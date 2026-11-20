"""Hand-rolled safetensors reader: 8-byte LE header length, JSON header, raw little-endian bytes."""
import json, struct
from pathlib import Path
import numpy as np

# safetensors dtype -> (numpy storage dtype, logical dtype name the backend understands)
_DT = {"BF16": (np.uint16, "bf16"), "F16": (np.float16, "f16"), "F32": (np.float32, "f32"),
       "I32": (np.int32, "i32"), "U32": (np.uint32, "u32"), "U8": (np.uint8, "u8")}


def read_safetensors(path):
    """Yield (name, numpy_view, logical_dtype). Views are zero-copy into a memmap."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    (n,) = struct.unpack("<Q", raw[:8].tobytes())
    header = json.loads(raw[8:8 + n].tobytes())
    base = 8 + n
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        np_dt, logical = _DT[meta["dtype"]]
        s, e = meta["data_offsets"]
        yield name, raw[base + s:base + e].view(np_dt).reshape(meta["shape"]), logical


def load_dir(model_dir):
    model_dir = Path(model_dir)
    out = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        for name, arr, dt in read_safetensors(f):
            out[name] = (arr, dt)
    return out


def resolve(model):
    """HF repo id or local dir -> local dir (uses the HF cache; downloads if missing)."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download
    try:  # cached copy first: no network round-trip
        return Path(snapshot_download(model, local_files_only=True))
    except Exception:
        return Path(snapshot_download(model))
