"""Verify a downloaded safetensors file before trusting it.

The torch wheel arrived at exactly the right byte count and was still corrupt:
a resume had stitched together the head of one build and the tail of another.
Size alone is not integrity. This parses the safetensors header, checks the
declared tensor extents against the actual file length, and confirms the MLM
head is present -- the three ways this particular file could be wrong.
"""

import argparse
import json
import struct
import sys
from pathlib import Path


def main(path: str) -> int:
    """Validate the structure and declared extents of a safetensors file.

    Args:
        path: Path to the safetensors model file.

    Returns:
        Zero when validation succeeds, otherwise one.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        print(f"FAIL: model file not found: {p}")
        return 1

    if size < 8:
        print(f"FAIL: file is too short for an 8-byte header: {size} bytes")
        return 1

    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > size - 8:
            print(f"FAIL: header length {n} implausible for {size}-byte file")
            return 1
        try:
            hdr = json.loads(f.read(n))
        except Exception as exc:                       # noqa: BLE001
            print(f"FAIL: header is not valid JSON ({exc}) -- truncated or "
                  f"spliced download")
            return 1

    if not isinstance(hdr, dict):
        print("FAIL: header JSON must be an object")
        return 1

    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    if not tensors:
        print("FAIL: header contains no tensors")
        return 1

    end = 8 + n
    data_size = size - end
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            print(f"FAIL: tensor {name!r} metadata must be an object")
            return 1
        offsets = tensor.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)):
            print(f"FAIL: tensor {name!r} has invalid data_offsets")
            return 1
        start, stop = offsets
        if start < 0 or stop < start or stop > data_size:
            print(f"FAIL: tensor {name!r} has out-of-bounds data_offsets")
            return 1

    maxoff = max(v["data_offsets"][1] for v in tensors.values())
    print(f"tensors: {len(tensors)}")
    print(f"file size: {size:,}  data region ends at: {end + maxoff:,}")
    if end + maxoff != size:
        print(f"FAIL: declared data ends at {end + maxoff:,} but file is "
              f"{size:,} -- {abs(size - end - maxoff):,} bytes off")
        return 1

    mlm = [k for k in tensors if "lm_head" in k or "predictions" in k]
    if not mlm:
        print("FAIL: no MLM head tensors -- pseudo-log-likelihood impossible")
        return 1
    print(f"MLM head present: {sorted(mlm)[:4]}")
    print("OK: header parses, extents match file length, MLM head present")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a downloaded safetensors model file.")
    parser.add_argument("path", help="path to the model.safetensors file")
    args = parser.parse_args()
    sys.exit(main(args.path))
