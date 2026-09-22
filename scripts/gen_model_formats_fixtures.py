#!/usr/bin/env python3
"""Write the dtype fixtures under `tests/fixtures/model-formats/`.

`tiny_mixed.safetensors` stores the same six numbers twice — once fp32, once
bfloat16 — because that is the only oracle widening can be checked against
without importing a reference: the values are exactly representable in bf16, so
a widened tensor has to come back bit-identical to the fp32 one, and any other
outcome is a bug in the shift. `third` is the one value that is not exact; it
pins the widened result to the right neighbourhood instead of to a bit pattern.

`tiny_f16.safetensors` exists for the opposite reason: fp16 is *not* bf16, and
the loader is supposed to say so by name rather than read it as if it were.

Run: `python3 scripts/gen_model_formats_fixtures.py`
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "tests" / "fixtures" / "model-formats"

# Exactly representable in bfloat16: widening them has to be a no-op.
EXACT = [1.0, -2.0, 0.5, 0.25, 1024.0, -0.125]


def bf16_bits(value: float) -> int:
    """Round-to-nearest-even fp32 -> bf16, as the reference toolchain does."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16) & 0xFFFF


def f32_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", v) for v in values)


def bf16_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<H", bf16_bits(v)) for v in values)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """`tensors`: name -> (dtype, shape, payload), written in insertion order."""
    header: dict[str, object] = {}
    body = b""
    cursor = 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
        body += payload
    meta = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The header is padded to a multiple of eight so the payload is aligned.
    padded = meta + b" " * (-len(meta) % 8)
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + body)
    print("wrote %s (%d bytes, %d tensors)" % (path, path.stat().st_size, len(tensors)))


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    write_safetensors(
        DEST / "tiny_mixed.safetensors",
        {
            "f32_row": ("F32", [len(EXACT)], f32_bytes(EXACT)),
            "bf16_row": ("BF16", [len(EXACT)], bf16_bytes(EXACT)),
            "third": ("BF16", [1], bf16_bytes([1.0 / 3.0])),
        },
    )
    write_safetensors(
        DEST / "tiny_f16.safetensors",
        {"half": ("F16", [2], struct.pack("<HH", 0x3C00, 0x4000))},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
