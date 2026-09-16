"""Repro/verify for generate.py line ~116: `or`-chained output fallback.

Old: frames = getattr(out,'frames',None) or getattr(out,'images',None) or ...
  -> bool(numpy_array) raises "truth value of an array with more than one
     element is ambiguous" when .frames is a multi-element array.

New: explicit `is not None` checks + container normalisation.
This test uses pure-stdlib fakes (no torch/numpy/GPU needed).
"""
import sys
import types


class FakeAmbigArray(list):
    """Mimics numpy/torch multi-element truthiness: bool() raises."""

    def __bool__(self):
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous. "
            "Use a.any() or a.all()"
        )


class FakeOut:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def old_extract(out):
    return getattr(out, "frames", None) or getattr(out, "images", None) or getattr(out, "videos", None)


def new_extract(out):
    # Mirror of generate.py logic (structure-only check, no export).
    frames_attr = getattr(out, "frames", None)
    if frames_attr is None:
        frames_attr = getattr(out, "images", None)
    if frames_attr is None:
        frames_attr = getattr(out, "videos", None)
    if frames_attr is None and isinstance(out, dict):
        for key in ("frames", "images", "videos"):
            if out.get(key) is not None:
                frames_attr = out.get(key)
                break
    return frames_attr


def main():
    fails = []

    # 1. Repro: .frames is a multi-element ambiguous array -> old code raises.
    out = FakeOut(frames=FakeAmbigArray(["f1", "f2", "f3"]))
    try:
        old_extract(out)
        fails.append("old code did NOT raise on ambiguous array (unexpected)")
    except ValueError as e:
        assert "ambiguous" in str(e), e
        print("PASS repro: old `or` chain raises ValueError(ambiguous) as reported")

    # 2. New code handles the same object (no truthiness check).
    got = new_extract(out)
    assert got is out.frames and len(got) == 3, got
    print("PASS new: ambiguous .frames extracted without bool()")

    # 3. Fallback order: frames None -> images used; videos used only if both None.
    o2 = FakeOut(frames=None, images=["img1"], videos=["vid1"])
    assert new_extract(o2) == ["img1"]
    o3 = FakeOut(frames=None, images=None, videos=["vid1"])
    assert new_extract(o3) == ["vid1"]
    o4 = FakeOut()  # nothing at all
    assert new_extract(o4) is None
    print("PASS new: frames > images > videos fallback order preserved")

    # 4. Dict-style output support.
    assert new_extract({"images": ["a"]}) == ["a"]
    assert new_extract({}) is None
    print("PASS new: dict outputs handled")

    # 5. Attribute present-but-None vs missing both fall through.
    o5 = FakeOut(frames=None)  # images/videos attrs missing entirely
    assert new_extract(o5) is None
    print("PASS new: missing attrs fall through cleanly")

    if fails:
        print("FAILURES:", fails)
        return 1
    print("ALL OUTPUT-FALLBACK CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
