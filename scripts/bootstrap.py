"""Bootstrap helper: download Parakeet model files and build the static decoder IR.

Run once on first start (from docker-entrypoint.sh) to populate /data with
everything the server needs. Idempotent — skips files that already exist.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


HF_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"

# Files we need locally. Some are pulled by onnx-asr's HF auto-download for the
# INT8 pipeline init, but we fetch all of them explicitly so the first run is
# deterministic and we don't depend on onnx-asr's download path resolving.
REQUIRED_FILES = [
    "config.json",
    "vocab.txt",
    "nemo128.onnx",                  # Mel preprocessor (~140 KB)
    "encoder-model.int8.onnx",       # INT8 encoder for onnx_asr pipeline init (~650 MB)
    "decoder_joint-model.int8.onnx", # INT8 decoder for onnx_asr pipeline init (~18 MB)
    "encoder-model.onnx",            # FP32 encoder graph (~41 MB)
    "encoder-model.onnx.data",       # FP32 encoder weights (~2.4 GB external)
    "decoder_joint-model.onnx",      # FP32 decoder/joint (~72 MB)
]


def download_if_missing(model_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    for fname in REQUIRED_FILES:
        target = model_dir / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"[bootstrap] Downloading {fname} from {HF_REPO} ...", flush=True)
        t0 = time.perf_counter()
        hf_hub_download(
            repo_id=HF_REPO,
            filename=fname,
            local_dir=str(model_dir),
        )
        print(
            f"[bootstrap]   done in {time.perf_counter()-t0:.1f}s "
            f"({target.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )


def build_static_decoder_ir(model_dir: Path, out_dir: Path) -> None:
    """Static-reshape the FP32 decoder/joint to fixed shapes and save as OV IR."""
    out_xml = out_dir / "decoder-static.xml"
    if out_xml.exists() and out_xml.stat().st_size > 0:
        print(f"[bootstrap] Static decoder IR exists at {out_xml}; skipping", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("[bootstrap] Building static decoder IR ...", flush=True)
    import openvino as ov  # imported here so docker layer ordering doesn't matter

    src = model_dir / "decoder_joint-model.onnx"
    if not src.exists():
        print(f"[bootstrap] FATAL: {src} not found", file=sys.stderr)
        sys.exit(1)

    core = ov.Core()
    model = core.read_model(str(src))
    # Static shapes match the per-token decoder loop exactly: one frame of
    # encoder output, one previous token, two LSTM states.
    model.reshape({
        "encoder_outputs": [1, 1024, 1],
        "targets": [1, 1],
        "target_length": [1],
        "input_states_1": [2, 1, 640],
        "input_states_2": [2, 1, 640],
    })
    ov.save_model(model, str(out_xml))
    print(f"[bootstrap]   saved {out_xml}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/data"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = data_dir / "nemo-parakeet-tdt-0.6b-v3"
    decoder_dir = data_dir / "static_decoder"

    download_if_missing(model_dir)
    build_static_decoder_ir(model_dir, decoder_dir)
    print("[bootstrap] All assets ready.", flush=True)


if __name__ == "__main__":
    main()
