"""Collected regression tests for batched autoregressive decoding."""

import os
import sys
import types
from contextlib import contextmanager

import pytest
import torch
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reasyn.sampler.sampler as sampler_module
from reasyn.chem.featurize import TokenType
from reasyn.chem.mol import Molecule
from reasyn.models.reasyn import ReaSyn
from reasyn.sampler.sampler import Sampler
from reasyn.utils.sample_utils import PredictResult, State


CKPT = os.environ.get(
    "REASYN_AR_CKPT",
    "/workspace/shared/ckpt/nv-reasyn-ar-166m-v2.ckpt",
)
TARGET = "C[NH+](C=C1SC(=O)NC1=S)C1(CO)CCC(OCC2CCCCC2)C1"
TEMPERATURE = 0.1
TOPK = 16


@contextmanager
def controlled_math():
    """Make batch-shape comparisons use the same arithmetic path."""
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


@pytest.fixture(scope="module")
def ar_context():
    if not torch.cuda.is_available():
        pytest.skip("AR checkpoint test requires CUDA")
    if not os.path.exists(CKPT):
        pytest.skip(f"AR checkpoint not found: {CKPT}")

    checkpoint = torch.load(CKPT, map_location="cpu")
    cfg = OmegaConf.create(checkpoint["hyper_parameters"]["config"])
    model = ReaSyn(cfg.model).cuda()
    model.load_state_dict({k[6:]: v for k, v in checkpoint["state_dict"].items()})
    model.eval()
    assert model.max_len >= 512

    smiles = Molecule(TARGET).tokenize_csmiles()[None].cuda()
    with torch.inference_mode():
        code, code_padding_mask = model.encoder(smiles)

    sampler = Sampler.__new__(Sampler)
    sampler.model = model
    sampler.device = torch.device("cuda")
    sampler._bucket_ar = True
    sampler._use_bf16 = False
    sampler._fpindex = object()
    sampler._rxn_matrix = object()
    return sampler, code, code_padding_mask


def make_prefix(length: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    prefix = torch.randint(4, int(TokenType.RXN_MIN), (length,), generator=generator).tolist()
    prefix[0] = int(TokenType.START)
    return prefix


def reaction_order(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, int(TokenType.RXN_MIN):int(TokenType.RXN_MAX) + 1].topk(TOPK).indices


@pytest.mark.parametrize("lmax", [1, 63, 64, 65, 511, 512])
@torch.inference_mode()
def test_production_forward_matches_per_row(ar_context, monkeypatch, lmax):
    sampler, code, code_padding_mask = ar_context
    mid = max(1, lmax // 2)
    lengths = sorted({1, mid, lmax})
    prefixes = {length: make_prefix(length, 1000 + lmax + length) for length in lengths}

    with controlled_math():
        reference = {
            length: sampler.model.sample(
                code=code,
                code_padding_mask=code_padding_mask,
                tokens=torch.tensor(prefix, device=sampler.device)[None],
                token_padding_mask=None,
            )[0].float()
            for length, prefix in prefixes.items()
        }

        original_sample = sampler.model.sample
        production_calls = []

        def record_sample(**kwargs):
            production_calls.append(kwargs)
            return original_sample(**kwargs)

        monkeypatch.setattr(sampler.model, "sample", record_sample)

        for batch_size in (1, 3, 9):
            if batch_size == 1:
                row_lengths = [lmax]
            else:
                row_lengths = ([1, mid, lmax] * 3)[:batch_size]
            seqs = [prefixes[length] for length in row_lengths]
            ref = torch.stack([reference[length] for length in row_lengths])
            batched = sampler._forward_ar_batched(code, code_padding_mask, seqs)

            call = production_calls[-1]
            expected_batch = 1 << max(0, batch_size - 1).bit_length()
            expected_length = min(((lmax + 63) // 64) * 64, sampler.model.max_len)
            assert call["token_padding_mask"] is None
            assert tuple(call["tokens"].shape) == (expected_batch, expected_length)

            ref_probs = torch.softmax(ref / TEMPERATURE, dim=-1)
            batched_probs = torch.softmax(batched / TEMPERATURE, dim=-1)
            tv = 0.5 * (ref_probs - batched_probs).abs().sum(dim=-1)
            assert tv.max().item() < 1e-4

            ref_top = reaction_order(ref)
            batched_top = reaction_order(batched)
            assert torch.equal(batched_top, ref_top)


@torch.inference_mode()
def test_predict_ar_batched_uses_production_reaction_order(ar_context, monkeypatch):
    sampler, code, code_padding_mask = ar_context
    seqs = [make_prefix(length, 2000 + length) for length in (1, 63, 65)]

    def force_reaction(probs, num_samples):
        return torch.full(
            (probs.shape[0], num_samples),
            int(TokenType.RXN_MIN),
            dtype=torch.long,
            device=probs.device,
        )

    monkeypatch.setattr(torch, "multinomial", force_reaction)
    monkeypatch.setattr(
        sampler_module,
        "get_reactions",
        lambda logits, rxn_matrix: logits.topk(TOPK).indices.tolist(),
    )

    with controlled_math():
        expected = reaction_order(sampler._forward_ar_batched(code, code_padding_mask, seqs))
        results = sampler._predict_ar_batched(
            code,
            code_padding_mask,
            seqs,
            sampling_direction="td",
        )

    assert [result.sampled_type for result in results] == ["RXN"] * len(seqs)
    assert [result.sampled_item for result in results] == expected.tolist()


def test_predict_ar_batched_lane_control_flow(monkeypatch):
    sampler = Sampler.__new__(Sampler)
    sampler.model = types.SimpleNamespace(max_len=8)
    sampler._fpindex = object()
    sampler._rxn_matrix = object()
    vocab_size = int(max(TokenType))

    base_lengths = {10: 2, 11: 3, 12: 2, 13: 6}

    def forward(self, code, code_padding_mask, seqs):
        logits = torch.full((len(seqs), vocab_size), -1000.0)
        for row, seq in enumerate(seqs):
            marker, length = seq[0], len(seq)
            if marker == 10:
                token = 4 if length == base_lengths[marker] else int(TokenType.MOL_END)
            elif marker == 11:
                token = 4 if length == 3 else 5 if length == 4 else int(TokenType.MOL_END)
            elif marker == 12:
                token = int(TokenType.MOL_END)
            else:
                token = 6
            logits[row, token] = 1000.0
        return logits

    sampler._forward_ar_batched = types.MethodType(forward, sampler)
    decoded = []
    monkeypatch.setattr(
        sampler_module,
        "decode_smiles",
        lambda tokens: decoded.append([int(token) for token in tokens]) or "ok",
    )
    monkeypatch.setattr(sampler_module, "get_reactants", lambda *args, **kwargs: ["reactant"])

    results = sampler._predict_ar_batched(
        None,
        None,
        [
            [10, int(TokenType.MOL_START)],
            [11, 7, int(TokenType.MOL_START)],
            [12, int(TokenType.MOL_START)],
            [13, 4, 4, 4, 4, int(TokenType.MOL_START)],
        ],
    )

    assert [result.sampled_type for result in results] == ["BB"] * 4
    assert decoded == [[4, 3], [4, 5, 3], [3], [6]]


class RecordingLock:
    def __init__(self):
        self.acquire_calls = 0
        self.release_calls = 0
        self.locked = False

    def acquire(self):
        self.acquire_calls += 1
        self.locked = True

    def release(self):
        self.release_calls += 1
        self.locked = False


@pytest.mark.parametrize(
    ("deadline_answers", "expected_acquires", "expected_releases"),
    [([True], 0, 0), ([False, True], 1, 1)],
)
def test_expired_deadline_skips_batched_prediction(
    deadline_answers,
    expected_acquires,
    expected_releases,
):
    sampler = Sampler.__new__(Sampler)
    state = State()
    sampler._active = [state]
    sampler._batched_ar = True
    lock = RecordingLock()
    answers = iter(deadline_answers)
    time_limit = types.SimpleNamespace(exceeded=lambda: next(answers))

    sampler._predict_ar_batched = lambda *args, **kwargs: pytest.fail(
        "batched prediction ran after the deadline"
    )
    sampler._evolve_ar_singlestep(
        gpu_lock=lock,
        time_limit=time_limit,
        sampling_direction="td",
    )

    assert lock.acquire_calls == expected_acquires
    assert lock.release_calls == expected_releases
    assert not lock.locked
    assert sampler._active == [state]


def test_legacy_deadline_check_remains_per_state():
    sampler = Sampler.__new__(Sampler)
    sampler._active = [State(), State()]
    sampler._finished = []
    sampler._aborted = []
    sampler._batched_ar = False
    sampler._factor = 1
    sampler._max_active_states = 2
    sampler.device = torch.device("cpu")
    sampler.__dict__["code"] = (None, None)
    lock = RecordingLock()
    answers = iter([False, True])
    time_limit = types.SimpleNamespace(exceeded=lambda: next(answers))
    calls = []

    def predict(self, **kwargs):
        calls.append(kwargs["tokens"].tolist())
        return PredictResult("ABORTED", None)

    sampler._predict_ar = types.MethodType(predict, sampler)
    sampler._evolve_ar_singlestep(
        gpu_lock=lock,
        time_limit=time_limit,
        sampling_direction="td",
    )

    assert calls == [[int(TokenType.START)]]
    assert lock.acquire_calls == lock.release_calls == 1
    assert not lock.locked


def test_batched_evolve_keeps_prefixes_on_cpu_lists():
    sampler = Sampler.__new__(Sampler)
    sampler._active = [State()]
    sampler._finished = []
    sampler._aborted = []
    sampler._batched_ar = True
    sampler._factor = 1
    sampler._max_active_states = 1
    sampler.device = torch.device("cpu")
    sampler.__dict__["code"] = (None, None)
    captured = []

    def predict(self, code, code_padding_mask, tokens_list, sampling_direction):
        captured.extend(tokens_list)
        return [PredictResult("ABORTED", None)]

    sampler._predict_ar_batched = types.MethodType(predict, sampler)
    sampler._evolve_ar_singlestep(sampling_direction="bu")

    assert captured == [[int(TokenType.START), int(TokenType.MOL_START)]]
    assert all(isinstance(token, int) for token in captured[0])


def test_bf16_is_opt_in(monkeypatch):
    class FakeModel:
        def __init__(self, model_type):
            self.model_type = model_type

        def parameters(self):
            yield torch.empty(0)

    molecule = types.SimpleNamespace(
        tokenize_csmiles=lambda: torch.tensor([4]),
        csmiles="C",
    )
    models = [FakeModel("autoregressive"), FakeModel("editflow")]

    monkeypatch.delenv("REASYN_BF16", raising=False)
    sampler = Sampler(object(), object(), molecule, models)
    assert not sampler._use_bf16

    monkeypatch.setenv("REASYN_BF16", "1")
    sampler = Sampler(object(), object(), molecule, models)
    assert sampler._use_bf16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
