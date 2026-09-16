"""End-to-end check of generate.generate_clip output extraction (no GPU/deps).

Stubs torch / PIL / diffusers.utils before importing generate.py, then drives
generate_clip() with fake pipes returning each output shape. Asserts export
receives a proper non-empty video and meta num_frames_out matches.
Covers the line-116 `or`-chain ambiguous-truthiness crash.
"""
import sys
import types
from pathlib import Path


# ---- Fake torch ----
class FakeTensor:
    def __init__(self, shape, float_flag=False):
        self._shape = tuple(shape)
        self._float = float_flag

    def dim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def is_floating_point(self):
        return self._float

    def clamp(self, a, b):
        return self

    def to(self, dtype):
        return self

    def numpy(self):
        n = self._shape[1] if len(self._shape) == 5 else self._shape[0]
        return [f"frame{i}" for i in range(n)]

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, idx):
        if len(self._shape) == 5 and idx == 0:
            return FakeTensor(self._shape[1:], self._float)
        return f"row{idx}"


class FakeGenerator:
    def __init__(self, device="cpu"):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


torch_mod = types.ModuleType("torch")
torch_mod.Tensor = FakeTensor
torch_mod.Generator = FakeGenerator
torch_mod.uint8 = "uint8"
cuda_mod = types.ModuleType("torch.cuda")
cuda_mod.is_available = lambda: False
torch_mod.cuda = cuda_mod
sys.modules["torch"] = torch_mod
sys.modules["torch.cuda"] = cuda_mod

# ---- Fake PIL ----
pil_mod = types.ModuleType("PIL")
pil_img_mod = types.ModuleType("PIL.Image")
pil_img_mod.open = lambda p: (_ for _ in ()).throw(AssertionError("no image expected"))
pil_mod.Image = pil_img_mod
sys.modules["PIL"] = pil_mod
sys.modules["PIL.Image"] = pil_img_mod

# ---- Fake diffusers.utils (export capture) ----
captured = {}
diff_mod = types.ModuleType("diffusers")
diff_utils_mod = types.ModuleType("diffusers.utils")


def fake_export(video, path, fps=None, quality=None, bitrate=None, macro_block_size=None, **kw):
    captured["video"] = video
    captured["path"] = path
    captured["fps"] = fps
    captured["quality"] = quality
    captured["macro_block_size"] = macro_block_size
    # len() only — never truthiness (video may be ambiguous-truthiness array)
    assert len(video) > 0, "export got empty video"


diff_utils_mod.export_to_video = fake_export
diff_mod.utils = diff_utils_mod
sys.modules["diffusers"] = diff_mod
sys.modules["diffusers.utils"] = diff_utils_mod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate  # noqa: E402


class AmbigBatch(list):
    """Outer container whose bool() raises, like a multi-element ndarray."""

    def __bool__(self):
        raise ValueError("The truth value of an array with more than one element is ambiguous")


class FakeOut:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def run_case(name, out, expect_frames, tmp):
    captured.clear()
    pipe = lambda **kw: out  # noqa: E731
    meta = generate.generate_clip(pipe, "prompt", tmp / f"{name}.mp4", seed=0)
    got = captured["video"]
    assert list(got) == list(expect_frames), f"{name}: export got {got!r}"
    assert meta["num_frames_out"] == len(expect_frames), f"{name}: meta {meta}"
    print(f"PASS {name}: export got {len(expect_frames)} frames, meta={meta}")


def main(tmpdir="outputs/_test_extract"):
    tmp = Path(__file__).resolve().parent / tmpdir
    tmp.mkdir(parents=True, exist_ok=True)
    frames3 = ["f1", "f2", "f3"]

    # 1. Typical diffusers batch-of-1 with ambiguous outer container
    run_case("batched_frames", FakeOut(frames=AmbigBatch([frames3])), frames3, tmp)
    # 2. Flat frame sequence already (no batch dim)
    run_case("flat_frames", FakeOut(frames=["a", "b"]), ["a", "b"], tmp)
    # 3-4. Fallback order images -> videos
    run_case("images_fallback", FakeOut(frames=None, images=[frames3]), frames3, tmp)
    run_case("videos_fallback", FakeOut(frames=None, images=None, videos=[frames3]), frames3, tmp)
    # 5. Dict-style output
    run_case("dict_frames", {"frames": [frames3]}, frames3, tmp)
    # 6. Torch 5D video tensor (B,T,H,W,C)
    run_case("torch5d", FakeOut(frames=FakeTensor((1, 4, 8, 8, 3))), [f"frame{i}" for i in range(4)], tmp)

    # 7. Empty container -> RuntimeError (not silent bad export)
    try:
        generate.generate_clip(lambda **kw: FakeOut(frames=[]), "p", tmp / "empty.mp4")
        print("FAIL empty: no error raised")
        return 1
    except RuntimeError as e:
        print(f"PASS empty: RuntimeError({e})")

    # 8. No usable attr -> RuntimeError
    try:
        generate.generate_clip(lambda **kw: FakeOut(), "p", tmp / "none.mp4")
        print("FAIL none: no error raised")
        return 1
    except RuntimeError as e:
        print(f"PASS none: RuntimeError({e})")

    print("ALL GENERATE_CLIP EXTRACTION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
