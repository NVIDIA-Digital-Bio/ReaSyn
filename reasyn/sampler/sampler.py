# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import copy
from collections.abc import Iterable
from functools import cached_property
from multiprocessing.synchronize import Lock
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.chem.mol import FingerprintOption, Molecule
from reasyn.data.collate import apply_collate, collate_tokens
from reasyn.data.common import featurize_stack
from reasyn.chem.featurize import decode_smiles, decode_tokens, TokenType
from reasyn.models.reasyn import ReaSyn
from reasyn.utils.editflow_utils import apply_ins_del_operations, get_adaptive_h
from reasyn.utils.sample_utils import get_reactants, get_reactions, \
    TimeLimit, PredictResult, State, ProductInfo, action_string_to_stack, get_sub_stacks


RXN_PATTERN = re.compile('R\d+')
MAX_CUDAGRAPH_CACHE_SIZE = 3


class Sampler:
    def __init__(
        self,
        fpindex: FingerprintIndex,
        rxn_matrix: ReactantReactionMatrix,
        mol: Molecule,
        model: list[ReaSyn, ReaSyn],
        factor: int = 16,
        max_active_states: int = 256,
        exact_break: bool = True,
        mols_to_filter=None,
        filter_sim: float = 0.8
    ) -> None:
        super().__init__()
        self._fpindex = fpindex
        self._rxn_matrix = rxn_matrix

        self.model, self.model_editflow = model
        assert self.model.model_type == 'autoregressive'
        assert self.model_editflow.model_type == 'editflow'
        self.device = next(iter(self.model.parameters())).device
        
        self._mol = mol
        smiles = mol.tokenize_csmiles()
        self._smiles = smiles[None].to(self.device)
        
        self._factor = factor
        self._max_active_states = max_active_states
        self.exact_break = exact_break
        self._mols_to_filter = mols_to_filter
        self._filter_sim = filter_sim
        # Batched AR decoding (env-toggleable for A/B against the original per-state loop).
        # REASYN_BATCHED_AR=0 -> original one-state-at-a-time path; REASYN_BF16=1 -> bf16 forward.
        self._batched_ar = os.environ.get('REASYN_BATCHED_AR', '1') != '0'
        self._use_bf16 = os.environ.get('REASYN_BF16', '0') != '0'
        # Pad AR forwards to power-of-2 batch / 64-multiple seq so cuBLAS/cuDNN reuse
        # kernels across the search's constantly-varying shapes (~7x fewer re-selections).
        self._bucket_ar = os.environ.get('REASYN_BUCKET', '1') != '0'
        # Capture a CUDA graph per bucket shape (encoder `code` is constant within a
        # Sampler, so it is baked in). Requires bucketing + bf16. Opt-in.
        self._cudagraph = (os.environ.get('REASYN_CUDAGRAPH', '0') == '1'
                           and self._batched_ar and self._bucket_ar and self._use_bf16
                           and self.device.type == 'cuda')
        self._graph_cache: dict = {}
        self._graph_stream = None

        # input filtering
        if self._mols_to_filter is not None and \
            max([self._mol.sim(f) for f in self._mols_to_filter] or [-1]) > self._filter_sim:
            print(f'Input molecule {self._mol.csmiles} filtered')
            quit()
        
        self._active: list[State] = [State()]
        self._finished: list[State] = []
        self._aborted: list[State] = []

    @cached_property
    def code(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            return self.model.encoder(self._smiles)
    
    @cached_property
    def code_editflow(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            return self.model_editflow.encoder(self._smiles)
        
    def _sort_states(self) -> None:
        self._active.sort(key=lambda s: s.score, reverse=True)
        self._active = self._active[: self._max_active_states]
    
    def _add_finished_states(self, finished) -> None:
        # remove failed states and num_steps=0 (just BB) states
        finished = [state for state in finished if state.stack.get_top()] # and state.stack.count_reactions()]
        for state in finished:
            state.scores = [max([p.sim(self._mol, FingerprintOption.morgan_for_tanimoto_similarity())
                                for p in state.stack.get_top()])]
        self._finished += finished
        # remove duplicated states
        self._finished = list({state.stack.get_action_string(): state for state in self._finished}.values())

    def _collate_editflow(self, feat_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        spec_tokens = {"tokens": collate_tokens}
        return apply_collate(spec_tokens, feat_list, self.model_editflow.max_len)
        
    def evolve(
        self,
        gpu_lock: Lock | None = None,
        time_limit: TimeLimit | None = None,
        max_evolve_steps: float = 8,
        num_cycles: int = 1,
        num_editflow_samples: int = 10,
        num_editflow_steps: int = 100,
    ) -> None:
        for i in range(num_cycles * 3):
            if time_limit is not None and time_limit.exceeded():
                break

            if self.exact_break:
                max_sim = max([state.score for state in self._finished] or [-1])
                if max_sim == 1.0:
                    break

            self._finished.sort(key=lambda s: s.score, reverse=True)
            
            if i % 3 == 2:  # EF
                if self._finished:
                    self._evolve_editflow(gpu_lock=gpu_lock, num_samples=num_editflow_samples)

            else:    # use BU and TD AR models alternatively
                sampling_direction = 'bu' if i % 3 == 0 else 'td'
                
                if self._finished:
                    finished = [state for state in self._finished if state.stack.count_reactions()] # at least one reaction
                    if finished:
                        scores = np.array([state.score for state in finished])
                        if scores.sum() == 0:
                            scores = np.ones_like(scores)
                        scores /= scores.sum()
                        self._active = []
                        for _ in range(self._max_active_states):
                            stack_to_repredict = np.random.choice(finished, p=scores).stack
                            sub_stack = get_sub_stacks(stack=stack_to_repredict,
                                                       num_samples=1, topdown=sampling_direction == 'td')[0]
                            self._active.append(State(sub_stack))
                    
                for _ in range(max_evolve_steps):
                    self._evolve_ar_singlestep(gpu_lock=gpu_lock, time_limit=time_limit,
                                               sampling_direction=sampling_direction)
                    if self.exact_break:
                        max_sim = max([state.score for state in self._finished] or [-1])
                        if max_sim == 1.0:
                            break
            
            self._finished.sort(key=lambda s: s.score, reverse=True)
            
    @torch.no_grad()
    def _predict_ar(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        tokens: torch.Tensor,
        temperature_token: float = 0.1,
        use_edit_distance: bool = True,
        sampling_direction: str = 'bu'
    ):
        def _sample_token(tokens, enforce_td=False):
            token_logits = self.model.sample(
                code=code,
                code_padding_mask=code_padding_mask,
                tokens=tokens,
                token_padding_mask=None)
            if enforce_td:
                token_logits[:, :TokenType.RXN_MIN] = -torch.inf     # RXN tokens only
            token_sampled = torch.multinomial(
                torch.nn.functional.softmax(token_logits / temperature_token, dim=-1),
                num_samples=1,
            )
            return token_sampled, token_logits

        assert len(tokens.shape) == 1, 'no batch allowed'
        
        if len(tokens) > self.model.max_len:
            sampled_type = 'ABORTED'
            sampled_item = None
        else:
            tokens = tokens[None, :]
            # for TD first token
            enforce_td = sampling_direction == 'td' and tokens.shape[-1] == 1
            token_sampled, token_logits = _sample_token(tokens, enforce_td=enforce_td)
            
        # latter condition for BU (TokenType.MOL_START is given at the start)
        if token_sampled == TokenType.MOL_START or tokens[:, -1] == TokenType.MOL_START:
            sampled_type = 'BB'
            token_sampled_bb = [token_sampled]
            while token_sampled != TokenType.MOL_END and tokens.shape[-1] < self.model.max_len - 2:
                tokens = torch.hstack([tokens, token_sampled])
                token_sampled, _ = _sample_token(tokens)
                token_sampled_bb.append(token_sampled)
            smiles = decode_smiles(token_sampled_bb)
            sampled_item = get_reactants(smiles, fpindex=self._fpindex, topk=100, use_edit_distance=use_edit_distance)
            if sampled_item is None:
                sampled_type = 'ABORTED'
        elif token_sampled >= TokenType.RXN_MIN:
            sampled_type = 'RXN'
            reaction_logits = token_logits[0, TokenType.RXN_MIN : TokenType.RXN_MAX + 1]    # (115,)
            sampled_item = get_reactions(reaction_logits, rxn_matrix=self._rxn_matrix)
        else:
            sampled_type = 'END'
            sampled_item = None
        
        return PredictResult(sampled_type, sampled_item)

    def _forward_ar_batched(self, code, code_padding_mask, seq_lists: list[list[int]]) -> torch.Tensor:
        """One batched decoder forward over CPU int-list prefixes. Right-pads on CPU,
        moves to device in a single H2D copy (no per-lane GPU<->CPU syncs), and gathers
        per-row last-real-token logits. Returns (B, vocab) in fp32.

        Shape bucketing (REASYN_BUCKET): the search feeds a new (batch, seq) shape on
        almost every call, so cuBLAS/cuDNN re-selects algorithms per shape (~7x overhead
        measured). Padding batch to a power-of-2 and seq to a 64-multiple collapses the
        distinct-shape count so kernels are reused. Extra rows/cols are ignored via gather
        + causal masking, so results are identical (the pad mask is redundant: the gathered
        last-real-token position is causally isolated from the right-padded future)."""
        device = self.device
        lens = [len(s) for s in seq_lists]
        B = len(seq_lists)
        Lmax = max(lens)
        if self._bucket_ar:
            Bp = 1 << max(0, (B - 1)).bit_length()                    # next power of two
            Lp = min(((Lmax + 63) // 64) * 64, self.model.max_len)    # next multiple of 64
        else:
            Bp, Lp = B, Lmax
        padded = [s + [0] * (Lp - len(s)) for s in seq_lists]
        padded += [[0] * Lp for _ in range(Bp - B)]                   # dummy rows -> ignored
        batch = torch.tensor(padded, dtype=torch.long, device=device)            # one H2D
        gather = torch.tensor([l - 1 for l in lens] + [0] * (Bp - B),
                              dtype=torch.long, device=device)

        if getattr(self, '_cudagraph', False):
            logits = self._run_ar_graph(code, code_padding_mask, batch, gather, Bp, Lp)
            with torch.cuda.device(self.device):
                return logits[:B].to(torch.float32, copy=True)

        codeB = code.expand(Bp, *code.shape[1:])
        maskB = code_padding_mask.expand(Bp, *code_padding_mask.shape[1:])
        if self._use_bf16:
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                logits = self.model.sample(code=codeB, code_padding_mask=maskB,
                                           tokens=batch, token_padding_mask=None,
                                           gather_idx=gather)
        else:
            logits = self.model.sample(code=codeB, code_padding_mask=maskB,
                                       tokens=batch, token_padding_mask=None,
                                       gather_idx=gather)
        return logits[:B].to(torch.float32, copy=True)

    def _run_ar_graph(self, code, code_padding_mask, batch, gather, Bp, Lp) -> torch.Tensor:
        """Replay a captured CUDA graph for this (Bp, Lp) shape. The encoder memory
        (code/mask) is constant within a Sampler so it is baked into the graph; only
        the token / gather-index static buffers are updated per call. Hot cached shapes
        amortize capture over the hundreds of forwards the search issues at that shape."""
        key = (Bp, Lp)
        e = self._graph_cache.pop(key, None)
        if e is not None:
            # Plain dicts preserve insertion order: reinsert hits for LRU eviction.
            self._graph_cache[key] = e
        if e is None:
            with torch.cuda.device(self.device):
                if len(self._graph_cache) >= MAX_CUDAGRAPH_CACHE_SIZE:
                    torch.cuda.synchronize(self.device)
                    oldest_key = next(iter(self._graph_cache))
                    oldest = self._graph_cache.pop(oldest_key)
                    self._reset_graph_entry(oldest)
                    torch.cuda.empty_cache()

                s_code = code.expand(Bp, *code.shape[1:])
                s_mask = code_padding_mask.expand(Bp, *code_padding_mask.shape[1:])
                s_tok = torch.zeros((Bp, Lp), dtype=torch.long, device=self.device)
                s_gi = torch.zeros((Bp,), dtype=torch.long, device=self.device)
                s_tok.copy_(batch); s_gi.copy_(gather)

                def _run():
                    return self.model.sample(code=s_code, code_padding_mask=s_mask,
                                             tokens=s_tok, token_padding_mask=None, gather_idx=s_gi)

                if self._graph_stream is None:
                    self._graph_stream = torch.cuda.Stream(device=self.device)

                # Torch 2.7 only documents shared-pool safety when replay order matches
                # capture order. Search order is arbitrary, so use independent pools and
                # cap them with the LRU above. Captures still share one device-bound stream.
                current = torch.cuda.current_stream(self.device)
                self._graph_stream.wait_stream(current)
                with torch.cuda.stream(self._graph_stream):
                    with torch.no_grad(), torch.autocast(
                            'cuda', dtype=torch.bfloat16, cache_enabled=False):
                        for _ in range(3):
                            _run()
                current.wait_stream(self._graph_stream)
                g = torch.cuda.CUDAGraph()
                with torch.no_grad(), torch.autocast(
                        'cuda', dtype=torch.bfloat16, cache_enabled=False), torch.cuda.graph(
                            g, stream=self._graph_stream):
                    s_out = _run()
                e = dict(g=g, s_tok=s_tok, s_gi=s_gi, s_out=s_out,
                         s_code=s_code, s_mask=s_mask)
                self._graph_cache[key] = e

        with torch.cuda.device(self.device):
            e['s_tok'].copy_(batch)
            e['s_gi'].copy_(gather)
            e['g'].replay()
        return e['s_out']

    @staticmethod
    def _reset_graph_entry(entry: dict) -> None:
        graph = entry.pop('g', None)
        if graph is not None:
            graph.reset()
        entry.clear()

    def _clear_graph_cache(self) -> None:
        """Release per-molecule graphs and return their pools to the CUDA allocator."""
        has_graph_state = bool(self._graph_cache) or self._graph_stream is not None
        if not has_graph_state:
            return
        if self.device.type != 'cuda':
            self._graph_cache.clear()
            self._graph_stream = None
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            while self._graph_cache:
                _, entry = self._graph_cache.popitem()
                self._reset_graph_entry(entry)
                del entry
            self._graph_stream = None
            # One graph-enabled worker owns each GPU, so returning unused cached
            # blocks here cannot disrupt another worker on the same device.
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _predict_ar_batched(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        tokens_list: list[list[int]],
        temperature_token: float = 0.1,
        use_edit_distance: bool = True,
        sampling_direction: str = 'bu',
    ) -> list[PredictResult]:
        """Batched equivalent of _predict_ar over all active states at once.

        Batches the GPU forwards (outer draw + inner BB-SMILES loop) while keeping the
        chemistry callbacks (get_reactants/get_reactions) per-state. Token bookkeeping
        stays on CPU as int-lists so each character-step incurs exactly one H2D (batch
        build) and one D2H (sample readout) sync regardless of batch size. Numerically
        equivalent to _predict_ar up to RNG-draw-order and bf16 (validated distributionally)."""
        MOL_START, MOL_END = int(TokenType.MOL_START), int(TokenType.MOL_END)
        RXN_MIN, RXN_MAX = int(TokenType.RXN_MIN), int(TokenType.RXN_MAX)
        max_len = self.model.max_len
        N = len(tokens_list)
        results: list[PredictResult | None] = [None] * N

        # CPU int-lists; immediate aborts (prefix over budget) excluded from the batch.
        act: list[int] = []
        seqs: list[list[int]] = []
        for k in range(N):
            L = len(tokens_list[k])
            if L > max_len:
                results[k] = PredictResult('ABORTED', None)
            else:
                act.append(k)
                seqs.append(tokens_list[k])
        if not act:
            return results

        def _sample(logits, td_first_idx=None) -> list[int]:
            if td_first_idx:
                logits = logits.clone()
                for i in td_first_idx:
                    logits[i, :RXN_MIN] = -torch.inf  # TD first token must be a reaction
            probs = torch.nn.functional.softmax(logits / temperature_token, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()  # one D2H

        # ---- Outer draw (one batched forward over all active states) ----
        logits0 = self._forward_ar_batched(code, code_padding_mask, seqs)  # (A, vocab)
        td_first_idx = [i for i, s in enumerate(seqs)
                        if sampling_direction == 'td' and len(s) == 1]
        tok0 = _sample(logits0, td_first_idx or None)  # list[int]

        # ---- Classify each active row (pure CPU) ----
        is_bb = [tok0[i] == MOL_START or seqs[i][-1] == MOL_START for i in range(len(seqs))]
        is_rxn = [(not is_bb[i]) and tok0[i] >= RXN_MIN for i in range(len(seqs))]

        # ---- Batched inner BB-SMILES loop (lane-masked, CPU token bookkeeping) ----
        bb_rows = [i for i in range(len(seqs)) if is_bb[i]]
        bb_tokens: dict[int, list[int]] = {i: [tok0[i]] for i in bb_rows}
        ctx: dict[int, list[int]] = {i: list(seqs[i]) for i in bb_rows}
        prev: dict[int, int] = {i: tok0[i] for i in bb_rows}

        def _cont(i):
            return prev[i] != MOL_END and len(ctx[i]) < max_len - 2

        active_bb = [i for i in bb_rows if _cont(i)]
        while active_bb:
            for i in active_bb:
                ctx[i].append(prev[i])  # append previous token to context
            logits = self._forward_ar_batched(code, code_padding_mask, [ctx[i] for i in active_bb])
            samp = _sample(logits)  # list[int]
            for j, i in enumerate(active_bb):
                prev[i] = samp[j]
                bb_tokens[i].append(samp[j])
            active_bb = [i for i in bb_rows if _cont(i)]

        # ---- Per-state chemistry (CPU): assemble PredictResult per active row ----
        for i in range(len(seqs)):
            k = act[i]
            if is_bb[i]:
                smiles = decode_smiles(bb_tokens[i])
                item = get_reactants(smiles, fpindex=self._fpindex, topk=100,
                                     use_edit_distance=use_edit_distance)
                results[k] = PredictResult('BB' if item is not None else 'ABORTED', item)
            elif is_rxn[i]:
                reaction_logits = logits0[i, RXN_MIN:RXN_MAX + 1]  # (115,)
                item = get_reactions(reaction_logits, rxn_matrix=self._rxn_matrix)
                results[k] = PredictResult('RXN', item)
            else:
                results[k] = PredictResult('END', None)

        return results

    def _evolve_ar_singlestep(
        self,
        gpu_lock: Lock | None = None,
        time_limit: TimeLimit | None = None,
        sampling_direction: str = 'bu'
    ) -> None:
        if len(self._active) == 0:
            return

        # Batched prediction is atomic: do not start it after the deadline.
        # The legacy path retains its original per-state deadline checks below.
        if self._batched_ar and time_limit is not None and time_limit.exceeded():
            return

        if gpu_lock is not None:
            gpu_lock.acquire()
            try:
                return self._evolve_ar_singlestep(
                    gpu_lock=None, time_limit=time_limit, sampling_direction=sampling_direction)
            finally:
                gpu_lock.release()

        feat_list = [
            featurize_stack(
                state.stack,
                end_token=False,
                sampling_direction=sampling_direction,
            )
            for state in self._active
        ]

        code, code_padding_mask = self.code

        # Build per-state token prefixes (with the BU MOL_START seed) once.
        tokens_list: list[list[int]] = []
        for feat in feat_list:
            tokens = feat['tokens'].tolist()
            if sampling_direction == 'bu' and len(tokens) == 1:  # for BU
                tokens.append(int(TokenType.MOL_START))
            tokens_list.append(tokens)

        # Batched path: one set of batched forwards for all active states.
        results_list = (
            self._predict_ar_batched(
                code=code, code_padding_mask=code_padding_mask,
                tokens_list=tokens_list, sampling_direction=sampling_direction)
            if self._batched_ar else None
        )

        finished: list[State] = []
        next: list[State] = []
        for k, base_state in enumerate(self._active):
            if results_list is None and time_limit is not None and time_limit.exceeded():
                break

            if results_list is not None:
                result = results_list[k]
            else:
                tokens = torch.tensor(tokens_list[k], dtype=torch.long, device=self.device)
                result = self._predict_ar(
                    code=code,
                    code_padding_mask=code_padding_mask,
                    tokens=tokens,
                    sampling_direction=sampling_direction,
                )
            sampled_type, sampled_item = result.sampled_type, result.sampled_item

            for i in range(self._factor):
                if sampling_direction == 'td':
                    if sampled_type == 'END':
                        finished.append(base_state)

                    elif sampled_type == 'BB' or sampled_type == 'RXN':
                        mol_or_rxn, idx, score = sampled_item[i]
                        new_state = copy.deepcopy(base_state)
                        new_state.stack.push_topdown(mol_or_rxn, idx)
                        new_state.scores.append(score)
                        next.append(new_state)

                    else:
                        self._aborted.append(base_state)

                else:
                    if sampled_type == 'END':
                        finished.append(base_state)

                    elif sampled_type == 'BB':
                        reactant, mol_idx, score = sampled_item[i]
                        new_state = copy.deepcopy(base_state)
                        new_state.stack.push_mol(reactant, mol_idx)
                        new_state.scores.append(score)
                        next.append(new_state)

                    elif sampled_type == 'RXN':
                        if i >= len(sampled_item):
                            self._aborted.append(new_state)
                            continue
                        new_state = copy.deepcopy(base_state)
                        for j in range(i, len(sampled_item)):  # try until success
                            reaction, rxn_idx, score = sampled_item[j]
                            success = new_state.stack.push_rxn(reaction, rxn_idx)
                            if success:
                                # intermediate filtering
                                filtered = False
                                if self._mols_to_filter is not None:
                                    for m in new_state.stack.get_top():
                                        if max([m.sim(f) for f in self._mols_to_filter]) > self._filter_sim:
                                            filtered = True
                                            print(f'Product molecule {m.csmiles} filtered')
                                            break
                                if not filtered:
                                    finished.append(new_state)
                                    new_state.scores.append(score)
                                    next.append(new_state)
                                    break
                        else:
                            j += 1
                            self._aborted.append(new_state)
                        sampled_item = sampled_item[:i] + sampled_item[j:]  # remove failed RXNs

                    else:
                        self._aborted.append(base_state)

        del self._active
        self._active = next
        self._sort_states()

        if sampling_direction == 'td':
            [state.stack.final_seq_topdown() for state in finished]
        self._add_finished_states(finished)

    @torch.inference_mode()
    def _predict_editflow(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        tokens: torch.Tensor,
        num_steps: int = 100,
        softmax_temp: float = 1.,
        use_edit_distance: bool = True,
    ):
        xt = tokens # x0
        decoded_all = [decode_tokens(xt[0])]
        
        delta_h = 1 / num_steps
        t = torch.zeros((tokens.shape[0], 1), device=tokens.device)
        
        for _ in range(num_steps):
            x_pad_mask = (xt == TokenType.END)
            ut, ins_logits, sub_logits = self.model_editflow.sample(
                code=code,
                code_padding_mask=code_padding_mask,
                tokens=xt,
                token_padding_mask=x_pad_mask,
                t=t,
            )
            
            ut = F.softplus(ut) # ensure positive rates
            ins_probs = F.softmax(ins_logits / softmax_temp, dim=-1)
            sub_probs = F.softmax(sub_logits / softmax_temp, dim=-1)
            lambda_ins = ut[:, :, 0]
            lambda_sub = ut[:, :, 1]
            lambda_del = ut[:, :, 2]
            adapt_h = get_adaptive_h(delta_h, t, self.model_editflow.scheduler)
            
            # Sample insertions and deletion/substitutions based on rates
            ins_mask = torch.rand(
                size=lambda_ins.shape, device=lambda_ins.device) < 1 - torch.exp(-adapt_h * lambda_ins)
            del_sub_mask = torch.rand(
                size=lambda_sub.shape, device=lambda_sub.device
            ) < 1 - torch.exp(-adapt_h * (lambda_sub + lambda_del))

            # For deletion/substitution, sample based on the relative rates
            prob_del = torch.where(
                del_sub_mask, lambda_del / (lambda_sub + lambda_del), torch.zeros_like(lambda_del))
            del_mask = torch.bernoulli(prob_del).bool()
            sub_mask = del_sub_mask & ~del_mask
            assert sub_mask.sum() + del_mask.sum() == del_sub_mask.sum()

            # Only sample tokens for non-pad positions, fill pad positions with TokenType.END
            ins_tokens = torch.full(ins_probs.shape[:2], TokenType.END, dtype=torch.long, device=ins_probs.device)
            sub_tokens = torch.full(sub_probs.shape[:2], TokenType.END, dtype=torch.long, device=sub_probs.device)
            non_pad_mask = ~x_pad_mask
            if non_pad_mask.any():
                ins_sampled = torch.multinomial(ins_probs[non_pad_mask], num_samples=1, replacement=True).squeeze(-1)
                sub_sampled = torch.multinomial(sub_probs[non_pad_mask], num_samples=1, replacement=True).squeeze(-1)
                ins_tokens[non_pad_mask] = ins_sampled
                sub_tokens[non_pad_mask] = sub_sampled

            # Apply operations based on masks
            xt[sub_mask] = sub_tokens[sub_mask]
            xt = apply_ins_del_operations(
                xt=xt,
                ins_mask=ins_mask,
                del_mask=del_mask,
                ins_tokens=ins_tokens,
                max_seq_len=self.model_editflow.max_len,
            )
            t = t + adapt_h
        decoded_all.extend([decode_tokens(x) for x in xt])
        decoded_all = list(set(decoded_all))
        
        finished = []
        for decoded in decoded_all:
            state = State()
            samples = decoded.split(',')
            for sample in samples:
                if RXN_PATTERN.match(sample):
                    rxn_idx = int(sample.lstrip('R'))
                    rxn = self._rxn_matrix.reactions[rxn_idx]
                    success = state.stack.push_rxn(rxn, rxn_idx)
                    if not success: break
                else:
                    sample = get_reactants(sample, fpindex=self._fpindex, topk=1, use_edit_distance=use_edit_distance)
                    if sample is None: break
                    sample = sample[0]
                    mol, mol_idx = sample.reactant, sample.index
                    state.stack.push_mol(mol, mol_idx)
            finished.append(state)
        return finished
    
    def _evolve_editflow(
        self,
        gpu_lock: Lock | None = None,
        num_samples: int = 1,
    ) -> None:
        scores = np.array([state.score for state in self._finished])
        if scores.sum() == 0:
            scores = np.ones_like(scores)
        scores /= scores.sum()
        states = [np.random.choice(self._finished, p=scores) for _ in range(num_samples)]
        feat_list = [
            featurize_stack(
                state.stack,
                end_token=True,
            )
            for state in states
        ]
        
        if gpu_lock is not None:
            gpu_lock.acquire()

        inputs = self._collate_editflow(feat_list)['tokens'].to(self.device)
        
        code, code_padding_mask = self.code_editflow
        code_size = list(code.size())
        code_size[0] = len(inputs)
        code = code.expand(code_size)
        mask_size = list(code_padding_mask.size())
        mask_size[0] = len(inputs)
        code_padding_mask = code_padding_mask.expand(mask_size)

        finished = self._predict_editflow(
            code=code,
            code_padding_mask=code_padding_mask,
            tokens=inputs,
        )
        self._add_finished_states(finished)

        if gpu_lock is not None:
            gpu_lock.release()
        
    def get_products(self) -> Iterable[ProductInfo]:
        visited: set[Molecule] = set()
        for state in self._finished:
            for mol in state.stack.get_top():
                if mol in visited:
                    continue
                yield ProductInfo(mol, state.stack)
                visited.add(mol)
        yield from []
    
    def get_dataframe(self, num_calc_extra_metrics: int = 10) -> pd.DataFrame:
        rows: list[dict[str, str | float]] = []
        smiles_to_mol: dict[str, Molecule] = {}
        for product in self.get_products():
            rows.append(
                {
                    "target": self._mol.smiles,
                    "smiles": product.molecule.smiles,
                    "score": self._mol.sim(product.molecule, FingerprintOption.morgan_for_tanimoto_similarity()),
                    "synthesis": product.stack.get_action_string(),
                    "num_steps": product.stack.count_reactions()
                }
            )
            smiles_to_mol[product.molecule.smiles] = product.molecule
        rows.sort(key=lambda r: r["score"], reverse=True)
        for row in rows[:num_calc_extra_metrics]:
            mol = smiles_to_mol[str(row["smiles"])]
            row["scf_sim"] = self._mol.scaffold.tanimoto_similarity(
                mol.scaffold,
                fp_option=FingerprintOption.morgan_for_tanimoto_similarity(),
            )
            row["pharm2d_sim"] = self._mol.dice_similarity(mol, fp_option=FingerprintOption.gobbi_pharm2d())
            row["rdkit_sim"] = self._mol.tanimoto_similarity(mol, fp_option=FingerprintOption.rdkit())

        df = pd.DataFrame(rows)
        return df
