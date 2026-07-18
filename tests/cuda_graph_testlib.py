"""Adversarial checkpoint-backed regression harness for CUDA-graph decoding.

Run on an A100 with the AR and Edit-Bridge checkpoint environment variables.
The pytest suite imports the focused checks from this file; the CLI also retains
the longer diagnostics used during review.
"""

import argparse
import gc
import multiprocessing as mp
import os
import pathlib
import queue
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reasyn.chem.featurize import TokenType
from reasyn.chem.mol import Molecule
from reasyn.models.reasyn import ReaSyn
from reasyn.sampler.sampler import MAX_CUDAGRAPH_CACHE_SIZE, Sampler
from reasyn.utils.sample_utils import State


AR_CKPT = os.environ.get(
    "REASYN_AR_CKPT", "/workspace/shared/ckpt/nv-reasyn-ar-166m-v2.ckpt"
)
EB_CKPT = os.environ.get(
    "REASYN_EB_CKPT", "/workspace/shared/ckpt/nv-reasyn-eb-174m-v2.ckpt"
)
DEV = "cuda"


def load_model(path: str, device: str = DEV) -> ReaSyn:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["hyper_parameters"]["config"])
    model = ReaSyn(cfg.model).to(device)
    model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
    model.eval()
    return model


def decoder_without_hints(
    decoder, code, code_padding_mask, tokens, token_padding_mask=None, t=None
):
    """The parent-branch Decoder.forward, kept literal for A/B comparison."""
    bsz, seqlen = tokens.size()
    x = decoder.embed(tokens)
    if t is not None and decoder.use_time_embed:
        x = x + decoder.te_dec(t)
    causal_mask = (
        torch.nn.Transformer.generate_square_subsequent_mask(
            x.size(1), dtype=x.dtype, device=x.device
        )
        if decoder.use_causal_mask
        else None
    )
    tgt_key_padding_mask = (
        torch.zeros((bsz, seqlen), dtype=x.dtype, device=x.device).masked_fill_(
            token_padding_mask, -torch.finfo(x.dtype).max
        )
        if token_padding_mask is not None
        else None
    )
    return decoder.dec(
        tgt=x,
        memory=code,
        tgt_mask=causal_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=code_padding_mask,
    )


def padded_inputs(model: ReaSyn, batch: int = 3, src_len: int = 40, tgt_len: int = 24):
    ntt = model.encoder.smiles_emb.num_embeddings
    smiles = torch.randint(1, ntt, (batch, src_len), device=DEV)
    for row, length in enumerate((src_len, src_len - 7, src_len - 13)[:batch]):
        smiles[row, length:] = 0
    code, code_mask = model.encoder(smiles)
    tokens = torch.randint(1, min(model.vocab_size, 100), (batch, tgt_len), device=DEV)
    lengths = torch.tensor((tgt_len, tgt_len - 5, tgt_len - 11)[:batch], device=DEV)
    token_mask = torch.arange(tgt_len, device=DEV)[None] >= lengths[:, None]
    tokens[token_mask] = 0
    return code, code_mask, tokens, token_mask


@torch.inference_mode()
def check_decoder(ar: ReaSyn, eb: ReaSyn):
    assert ar.model_type == "autoregressive" and ar.decoder.use_causal_mask
    assert eb.model_type == "editflow" and not eb.decoder.use_causal_mask

    for name, model in (("AR", ar), ("EB", eb)):
        code, code_mask, tokens, token_mask = padded_inputs(model)
        t = torch.full((tokens.size(0), 1), 0.37, device=DEV) if name == "EB" else None
        old = decoder_without_hints(model.decoder, code, code_mask, tokens, token_mask, t)
        new = model.decoder(code, code_mask, tokens, token_mask, t)
        diff = (old - new).abs()
        print(
            f"DECODER_{name} equal={torch.equal(old, new)} "
            f"max_abs={diff.max().item():.9g} padded=True"
        )
        assert torch.equal(old, new)

        if name == "EB":
            ut, ins, sub = model.sample(code, code_mask, tokens, token_mask, t)
            assert ut.shape[:2] == tokens.shape and ins.shape[:2] == tokens.shape
            assert sub.shape == ins.shape
            assert torch.isfinite(ut).all() and torch.isfinite(ins).all() and torch.isfinite(sub).all()
            print(f"EDIT_BRIDGE_SAMPLE_OK shapes={tuple(ut.shape)},{tuple(ins.shape)}")

    # A future-token perturbation must not affect an AR prefix, but must affect EB.
    for name, model in (("AR", ar), ("EB", eb)):
        code, code_mask, tokens, _ = padded_inputs(model)
        changed = tokens.clone()
        changed[:, 8:] = (changed[:, 8:] + 17) % min(model.vocab_size, 100)
        t = torch.full((tokens.size(0), 1), 0.37, device=DEV) if name == "EB" else None
        # Edit-Bridge's custom MHA requires the slow path used in production, where
        # _predict_editflow always supplies a token padding mask.
        semantic_mask = torch.zeros_like(tokens, dtype=torch.bool) if name == "EB" else None
        y1 = model.decoder(code, code_mask, tokens, semantic_mask, t)
        y2 = model.decoder(code, code_mask, changed, semantic_mask, t)
        prefix_delta = (y1[:, :8] - y2[:, :8]).abs().max().item()
        print(f"CAUSAL_SEMANTICS_{name} future_to_prefix_max_abs={prefix_delta:.9g}")
        if name == "AR":
            assert prefix_delta == 0.0
        else:
            assert prefix_delta > 0.0

    # On the tested runtime the implicit detector performs a scalar extraction.
    code, code_mask, tokens, token_mask = padded_inputs(ar)
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            decoder_without_hints(ar.decoder, code, code_mask, tokens, token_mask)
            torch.cuda.synchronize()
        sync_ops = {
            event.key: event.count
            for event in prof.key_averages()
            if "local_scalar" in event.key or "equal" in event.key
        }
        print(f"IMPLICIT_CAUSAL_DETECT_SYNC_OPS {sync_ops}")
    except Exception as exc:
        print(f"IMPLICIT_CAUSAL_DETECT_PROFILER_SKIPPED {type(exc).__name__}: {exc}")


def bare_sampler(model: ReaSyn, use_graph: bool) -> Sampler:
    sampler = Sampler.__new__(Sampler)
    sampler.model = model
    sampler.device = next(model.parameters()).device
    sampler._bucket_ar = True
    sampler._use_bf16 = True
    sampler._cudagraph = use_graph
    sampler._graph_cache = {}
    sampler._graph_stream = None
    return sampler


def production_sampler(model: ReaSyn, use_graph: bool) -> Sampler:
    """Construct through Sampler.__init__ with the real environment toggle."""
    class EditflowStub:
        model_type = "editflow"

    keys = ("REASYN_CUDAGRAPH", "REASYN_BATCHED_AR", "REASYN_BUCKET", "REASYN_BF16")
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(
            REASYN_CUDAGRAPH="1" if use_graph else "0",
            REASYN_BATCHED_AR="1",
            REASYN_BUCKET="1",
            REASYN_BF16="1",
        )
        return Sampler(None, None, Molecule("CCO"), [model, EditflowStub()])
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_sequences(seed: int, count: int, max_len: int = 60) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(max(1, max_len - 20), max_len + 1, (count,), generator=generator)
    return [
        torch.randint(1, 100, (int(length),), generator=generator).tolist()
        for length in lengths
    ]


def static_batch(seqs: list[list[int]], bp: int, lp: int, device):
    padded = [seq + [0] * (lp - len(seq)) for seq in seqs]
    padded += [[0] * lp for _ in range(bp - len(seqs))]
    batch = torch.tensor(padded, dtype=torch.long, device=device)
    gather = torch.tensor(
        [len(seq) - 1 for seq in seqs] + [0] * (bp - len(seqs)),
        dtype=torch.long,
        device=device,
    )
    return batch, gather


@torch.inference_mode()
def check_graph(ar: ReaSyn):
    torch.manual_seed(123)
    smiles1 = torch.randint(1, ar.encoder.smiles_emb.num_embeddings, (1, 48), device=DEV)
    smiles2 = torch.randint(1, ar.encoder.smiles_emb.num_embeddings, (1, 48), device=DEV)
    code1, mask1 = ar.encoder(smiles1)
    code2, mask2 = ar.encoder(smiles2)
    eager = production_sampler(ar, False)
    graph = production_sampler(ar, True)
    assert not eager._cudagraph and graph._cudagraph

    first_capture_s = None
    equal_cases = 0
    max_graph_eager = 0.0
    max_prob_delta = 0.0
    argmax_matches = 0
    sample_matches = 0
    repeated_draw_matches = 0
    repeated_draws = 0
    rows = 0
    for seed in range(20):
        # Counts 5..8 and lengths 40..60 all hit exactly cache key (8, 64).
        seqs = make_sequences(seed, 5 + seed % 4)
        ref = eager._forward_ar_batched(code1, mask1, seqs)
        start = time.perf_counter()
        got = graph._forward_ar_batched(code1, mask1, seqs)
        torch.cuda.synchronize()
        if first_capture_s is None:
            first_capture_s = time.perf_counter() - start
        max_abs = (ref - got).abs().max().item()
        equal_cases += int(torch.equal(ref, got))
        assert torch.equal(ref, got), f"graph/eager mismatch for same-shape seed {seed}"
        max_graph_eager = max(max_graph_eager, max_abs)
        ref_prob = torch.softmax(ref / 0.1, dim=-1)
        got_prob = torch.softmax(got / 0.1, dim=-1)
        max_prob_delta = max(max_prob_delta, (ref_prob - got_prob).abs().max().item())
        argmax_matches += int((ref.argmax(-1) == got.argmax(-1)).sum())
        state = torch.cuda.get_rng_state(graph.device)
        ref_sample = torch.multinomial(ref_prob, 1)
        torch.cuda.set_rng_state(state, graph.device)
        got_sample = torch.multinomial(got_prob, 1)
        assert torch.equal(ref_sample, got_sample)
        sample_matches += int((ref_sample == got_sample).sum())
        state = torch.cuda.get_rng_state(graph.device)
        ref_draws = torch.multinomial(ref_prob, 128, replacement=True)
        torch.cuda.set_rng_state(state, graph.device)
        got_draws = torch.multinomial(got_prob, 128, replacement=True)
        assert torch.equal(ref_draws, got_draws)
        repeated_draw_matches += int((ref_draws == got_draws).sum())
        repeated_draws += ref_draws.numel()
        rows += len(seqs)
    entry = graph._graph_cache[(8, 64)]
    assert entry["s_code"].stride(0) == 0
    assert entry["s_mask"].stride(0) == 0
    assert graph._graph_stream.device == graph.device
    print(
        f"GRAPH_REPLAY_EQ cases=20 bit_equal_cases={equal_cases}/20 "
        f"max_abs={max_graph_eager:.9g} max_prob_at_T0.1={max_prob_delta:.9g} "
        f"argmax_match={argmax_matches}/{rows} same_rng_sample_match={sample_matches}/{rows} "
        f"same_rng_repeated_draw_match={repeated_draw_matches}/{repeated_draws} "
        f"first_capture_s={first_capture_s:.6f} cache_keys={list(graph._graph_cache)}"
    )

    # Raw graph output aliases and is overwritten; public output must not.
    seqs1, seqs2 = make_sequences(77, 7), make_sequences(88, 7)
    batch1, gather1 = static_batch(seqs1, 8, 64, graph.device)
    batch2, gather2 = static_batch(seqs2, 8, 64, graph.device)
    raw1 = graph._run_ar_graph(code1, mask1, batch1, gather1, 8, 64)
    torch.cuda.synchronize()
    raw_snapshot = raw1.clone()
    raw2 = graph._run_ar_graph(code1, mask1, batch2, gather2, 8, 64)
    torch.cuda.synchronize()
    raw_overwritten = not torch.equal(raw1, raw_snapshot)
    assert raw1.data_ptr() == raw2.data_ptr() == entry["s_out"].data_ptr()
    assert raw_overwritten

    public1 = graph._forward_ar_batched(code1, mask1, seqs1)
    torch.cuda.synchronize()
    public_snapshot = public1.clone()
    public2 = graph._forward_ar_batched(code1, mask1, seqs2)
    torch.cuda.synchronize()
    assert torch.equal(public1, public_snapshot)
    assert public1.data_ptr() != entry["s_out"].data_ptr()
    print(
        f"ALIAS_TRAP raw_dtype={entry['s_out'].dtype} raw_reused=True "
        f"raw_overwritten={raw_overwritten} public_dtype={public1.dtype} "
        f"public_distinct_storage=True public_survived_replay=True "
        f"new_output_changed={not torch.equal(public1, public2)}"
    )

    # Deliberately violate the baked-code contract to expose the failure mode.
    seqs = make_sequences(99, 6)
    graph1 = graph._forward_ar_batched(code1, mask1, seqs)
    eager2 = eager._forward_ar_batched(code2, mask2, seqs)
    stale = graph._forward_ar_batched(code2, mask2, seqs)
    fresh_graph = production_sampler(ar, True)
    fresh = fresh_graph._forward_ar_batched(code2, mask2, seqs)
    torch.cuda.synchronize()
    assert torch.equal(stale, graph1)
    assert not torch.equal(stale, eager2)
    assert torch.equal(fresh, eager2)
    print(
        f"BAKED_CODE same_cache_ignores_new_code=True "
        f"stale_vs_new_max_abs={(stale-eager2).abs().max().item():.9g} "
        f"fresh_graph_vs_eager_max_abs={(fresh-eager2).abs().max().item():.9g} "
        f"fresh_graph_vs_eager_bit_equal=True cache_objects_distinct="
        f"{graph._graph_cache is not fresh_graph._graph_cache}"
    )

    # Independent graph pools must be safe under arbitrary serial replay order.
    # Keep a copied result alive while replaying the other shape to catch aliasing.
    seqs_64 = make_sequences(501, 7, 60)
    seqs_128 = make_sequences(502, 3, 100)
    for index in range(12):
        first_seqs, second_seqs = (
            (seqs_64, seqs_128) if index % 2 == 0 else (seqs_128, seqs_64)
        )
        expected = eager._forward_ar_batched(code1, mask1, first_seqs)
        saved = graph._forward_ar_batched(code1, mask1, first_seqs)
        saved_snapshot = saved.clone()
        graph._forward_ar_batched(code1, mask1, second_seqs)
        torch.cuda.synchronize()
        assert torch.equal(saved, expected)
        assert torch.equal(saved, saved_snapshot)
    pools = {entry["g"].pool() for entry in graph._graph_cache.values()}
    assert len(pools) == len(graph._graph_cache)
    print(
        f"PRIVATE_POOL_ALTERNATING pass=True keys={sorted(graph._graph_cache)} "
        f"pool_count={len(pools)} stream_device={graph._graph_stream.device}"
    )

    # Steady-state timing at the key used above, including Python/tensor copies.
    cases = [make_sequences(1000 + i, 8) for i in range(30)]
    for seqs in cases[:3]:
        eager._forward_ar_batched(code1, mask1, seqs)
        graph._forward_ar_batched(code1, mask1, seqs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for seqs in cases:
        eager._forward_ar_batched(code1, mask1, seqs)
    torch.cuda.synchronize()
    eager_s = time.perf_counter() - start
    start = time.perf_counter()
    for seqs in cases:
        graph._forward_ar_batched(code1, mask1, seqs)
    torch.cuda.synchronize()
    graph_s = time.perf_counter() - start
    speedup = eager_s / graph_s
    breakeven = first_capture_s / max(eager_s / len(cases) - graph_s / len(cases), 1e-12)
    print(
        f"SPEED key=(8,64) n=30 eager_s={eager_s:.6f} graph_s={graph_s:.6f} "
        f"speedup={speedup:.3f}x capture_s={first_capture_s:.6f} "
        f"estimated_replays_to_amortize={breakeven:.1f}"
    )

    # Exceed the cache limit, verify LRU eviction, then revisit an evicted shape.
    lru_cases = [
        make_sequences(601, 2, 180),
        make_sequences(602, 1, 240),
        make_sequences(603, 16, 60),
        make_sequences(604, 4, 100),
    ]
    for seqs in lru_cases:
        expected = eager._forward_ar_batched(code1, mask1, seqs)
        got = graph._forward_ar_batched(code1, mask1, seqs)
        assert torch.equal(got, expected)
        assert len(graph._graph_cache) <= MAX_CUDAGRAPH_CACHE_SIZE
    pools = {entry["g"].pool() for entry in graph._graph_cache.values()}
    assert len(graph._graph_cache) == MAX_CUDAGRAPH_CACHE_SIZE
    assert len(pools) == MAX_CUDAGRAPH_CACHE_SIZE
    assert list(graph._graph_cache) == [(1, 256), (16, 64), (4, 128)]
    print(
        f"GRAPH_LRU pass=True resident={len(graph._graph_cache)} "
        f"limit={MAX_CUDAGRAPH_CACHE_SIZE} keys={list(graph._graph_cache)}"
    )
    passed = equal_cases == 20
    graph._clear_graph_cache()
    fresh_graph._clear_graph_cache()
    return passed


def check_sampler_contract_and_guards(ar: ReaSyn, eb: ReaSyn):
    keys = ("REASYN_CUDAGRAPH", "REASYN_BATCHED_AR", "REASYN_BUCKET", "REASYN_BF16")
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(
            REASYN_CUDAGRAPH="1", REASYN_BATCHED_AR="1",
            REASYN_BUCKET="1", REASYN_BF16="1",
        )
        first = Sampler(None, None, Molecule("CCO"), [ar, eb])
        second = Sampler(None, None, Molecule("CCN"), [ar, eb])
        code_a = first.code
        assert code_a[0].data_ptr() == first.code[0].data_ptr()
        assert first._graph_cache is not second._graph_cache
        print(
            "SAMPLER_CONTRACT code_cached=True graph_cache_per_sampler=True "
            f"graph_enabled={first._cudagraph}"
        )

        os.environ["REASYN_BUCKET"] = "0"
        no_bucket = Sampler(None, None, Molecule("CCO"), [ar, eb])
        os.environ.update(REASYN_BUCKET="1", REASYN_BF16="0")
        no_bf16 = Sampler(None, None, Molecule("CCO"), [ar, eb])
        os.environ.update(REASYN_BF16="1", REASYN_BATCHED_AR="0")
        no_batched = Sampler(None, None, Molecule("CCO"), [ar, eb])
        assert not no_bucket._cudagraph and not no_bf16._cudagraph
        assert not no_batched._cudagraph
        print("GRAPH_GUARDS bucket0=False bf16_0=False batched0=False")

        class Dummy(torch.nn.Module):
            def __init__(self, model_type):
                super().__init__()
                self.model_type = model_type
                self.p = torch.nn.Parameter(torch.zeros(()))

        os.environ.update(REASYN_BUCKET="1", REASYN_BF16="1", REASYN_BATCHED_AR="1")
        cpu_sampler = Sampler(
            None,
            None,
            Molecule("CCO"),
            [Dummy("autoregressive"), Dummy("editflow")],
        )
        print(
            f"CPU_GUARD_ON_GPU_HOST graph_enabled={cpu_sampler._cudagraph} "
            f"model_device={cpu_sampler.device} cuda_available={torch.cuda.is_available()}"
        )
        assert not cpu_sampler._cudagraph
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@torch.inference_mode()
def run_core():
    print(
        f"RUNTIME torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name()} current_device={torch.cuda.current_device()}"
    )
    torch.manual_seed(7)
    ar = load_model(AR_CKPT)
    eb = load_model(EB_CKPT)
    check_decoder(ar, eb)
    check_sampler_contract_and_guards(ar, eb)
    graph_equal = check_graph(ar)
    print(f"CORE_GRAPH_EAGER_EQUIVALENCE pass={graph_equal}")
    assert graph_equal, "CUDA-graph output is not bit-identical to eager output"


@torch.inference_mode()
def run_growth(full: bool = False):
    ar = load_model(AR_CKPT)
    src_len = 256 if full else 64
    smiles = torch.randint(1, ar.encoder.smiles_emb.num_embeddings, (1, src_len), device=DEV)
    code, mask = ar.encoder(smiles)
    sampler = bare_sampler(ar, True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_alloc = torch.cuda.memory_allocated()
    baseline_res = torch.cuda.memory_reserved()
    shapes = (
        sorted(
            ((bp, lp) for bp in (1, 2, 4, 8, 16, 32, 64, 128, 256)
             for lp in (64, 128, 192, 256, 320, 384, 448, 512)),
            key=lambda shape: shape[0] * shape[1] ** 2,
        )
        if full
        else [
            (1, 64), (2, 64), (4, 64), (8, 64), (16, 64), (32, 64),
            (1, 128), (2, 128), (4, 128), (8, 128), (1, 256), (2, 256),
        ]
    )
    print(
        f"GROWTH_BASE allocated_mib={baseline_alloc/2**20:.1f} "
        f"reserved_mib={baseline_res/2**20:.1f}"
    )
    captured = 0
    for bp, lp in shapes:
        seqs = make_sequences(bp * 1000 + lp, bp, lp)
        start = time.perf_counter()
        try:
            sampler._forward_ar_batched(code, mask, seqs)
        except torch.OutOfMemoryError as exc:
            print(f"GROWTH_OOM key=({bp},{lp}) cache={len(sampler._graph_cache)} error={exc}")
            break
        captured += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(
            f"GROWTH key=({bp},{lp}) capture_s={elapsed:.4f} cache={len(sampler._graph_cache)} "
            f"allocated_mib={alloc/2**20:.1f} delta_alloc_mib={(alloc-baseline_alloc)/2**20:.1f} "
            f"reserved_mib={reserved/2**20:.1f} delta_reserved_mib={(reserved-baseline_res)/2**20:.1f}"
        )
    completed = captured == len(shapes)
    reserved_growth = torch.cuda.memory_reserved() - baseline_res
    print(
        f"GROWTH_COMPLETE pass={completed} captured={captured}/{len(shapes)} "
        f"resident={len(sampler._graph_cache)}/{MAX_CUDAGRAPH_CACHE_SIZE} "
        f"peak_allocated_mib={torch.cuda.max_memory_allocated()/2**20:.1f} "
        f"delta_reserved_mib={reserved_growth/2**20:.1f}"
    )
    assert completed, "shape-cache growth run did not capture every requested key"
    assert len(sampler._graph_cache) <= MAX_CUDAGRAPH_CACHE_SIZE
    assert reserved_growth < 12 * 2**30, "bounded graph cache exceeded 12 GiB"
    sampler._clear_graph_cache()
    gc.collect()
    torch.cuda.synchronize()
    print(
        f"GROWTH_AFTER_CLEAR allocated_mib={torch.cuda.memory_allocated()/2**20:.1f} "
        f"reserved_mib={torch.cuda.memory_reserved()/2**20:.1f}"
    )
    assert torch.cuda.memory_reserved() <= baseline_res + 512 * 2**20


def worker_capture(rank: int, lock, result_queue):
    try:
        torch.cuda.set_device(0)
        ar = load_model(AR_CKPT)
        smiles = torch.randint(1, ar.encoder.smiles_emb.num_embeddings, (1, 32), device=DEV)
        with torch.inference_mode():
            code, mask = ar.encoder(smiles)
            sampler = bare_sampler(ar, True)
            with lock:
                out = sampler._forward_ar_batched(code, mask, make_sequences(rank + 300, 4))
                torch.cuda.synchronize()
            result_queue.put((rank, "ok", float(out.sum()), len(sampler._graph_cache)))
    except Exception as exc:
        result_queue.put((rank, "error", f"{type(exc).__name__}: {exc}", 0))


def run_multiprocess():
    ctx = mp.get_context("spawn")
    lock = ctx.Lock()
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=worker_capture, args=(rank, lock, result_queue)) for rank in range(2)]
    for worker in workers:
        worker.start()
    results = []
    for _ in workers:
        try:
            results.append(result_queue.get(timeout=240))
        except queue.Empty:
            results.append((-1, "timeout", "no result", 0))
    for worker in workers:
        worker.join(timeout=30)
    print(f"MULTIPROCESS_RESULTS {sorted(results)} exitcodes={[w.exitcode for w in workers]}")
    assert all(item[1] == "ok" for item in results)
    assert all(worker.exitcode == 0 for worker in workers)
    print("MULTIPROCESS_PASS workers=2 captures=2 serialized_by_gpu_lock=True")


def run_lock_failure():
    class FailingModel:
        @staticmethod
        def encoder(_):
            raise RuntimeError("synthetic post-acquire failure")

    sampler = Sampler.__new__(Sampler)
    sampler._active = [State()]
    sampler._batched_ar = True
    sampler.model = FailingModel()
    sampler._smiles = torch.zeros((1, 1), dtype=torch.long)
    lock = mp.Lock()
    try:
        sampler._evolve_ar_singlestep(gpu_lock=lock)
    except RuntimeError as exc:
        print(f"LOCK_INJECTED_ERROR {exc}")
    reacquired = lock.acquire(block=False)
    print(f"LOCK_AFTER_EXCEPTION released={reacquired}")
    assert reacquired
    lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("core", "growth", "fullgrowth", "multiprocess", "lock")
    )
    args = parser.parse_args()
    {
        "core": run_core,
        "growth": run_growth,
        "fullgrowth": lambda: run_growth(full=True),
        "multiprocess": run_multiprocess,
        "lock": run_lock_failure,
    }[args.mode]()
