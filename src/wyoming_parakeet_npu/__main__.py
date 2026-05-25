"""Wyoming server for Parakeet TDT 0.6B v3 multilingual on Intel NPU.

Loads the model via onnx_asr, then replaces its encoder + decoder/joint
ORT InferenceSessions with OpenVINO NPU-compiled shims. The encoder uses
multi-bucket dispatch (one compiled blob per audio length, picked at request
time) with optional lazy loading for large buckets.

Designed for one specific use case: smart-home / dictation STT on Intel
Core Ultra CPUs with the AI Boost NPU. No model selection, no quantization
selection — just Parakeet TDT 0.6B v3.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from functools import partial

import onnx_asr
import onnxruntime
import openvino as ov
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import ParakeetEventHandler
from .shims import OpenVinoDecoderShim, OpenVinoEncoderShim

_LOGGER = logging.getLogger(__name__)

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
SUPPORTED_LANGUAGES = (
    "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "uk",
    "bg", "hr", "cs", "da", "et", "fi", "el", "hu", "lv", "lt",
    "mt", "ro", "sk", "sl", "sv",
)


def _parse_buckets(s: str | None) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()] if s else []


def _build_wyoming_info(default_language: str) -> Info:
    return Info(asr=[AsrProgram(
        name="parakeet-npu",
        description="Parakeet TDT 0.6B v3 multilingual on Intel NPU (OpenVINO)",
        attribution=Attribution(
            name="cibernox/wyoming-parakeet-on-intel-npu",
            url="https://github.com/cibernox/wyoming-parakeet-on-intel-npu",
        ),
        installed=True,
        version=__version__,
        models=[AsrModel(
            name=MODEL_NAME,
            description=f"Multilingual ASR (default: {default_language})",
            attribution=Attribution(
                name="NVIDIA NeMo (model) + onnx-asr (pipeline)",
                url="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx",
            ),
            installed=True,
            languages=list(SUPPORTED_LANGUAGES),
            version="0.1",
        )],
    )])


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wyoming-parakeet-npu",
        description="Wyoming STT server for Parakeet TDT on Intel NPU",
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10300"),
        help="Wyoming server URI (default: tcp://0.0.0.0:10300)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "/data"),
        help="Directory holding the model files and OpenVINO cache (default: /data)",
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("LANGUAGE", "en"),
        choices=SUPPORTED_LANGUAGES,
        help="Default transcription language when the client does not specify one",
    )
    parser.add_argument(
        "--encoder-buckets",
        default=os.environ.get("ENCODER_BUCKETS", "5"),
        help="Comma-separated EAGER bucket sizes in seconds (default: 5)",
    )
    parser.add_argument(
        "--encoder-lazy-buckets",
        default=os.environ.get("ENCODER_LAZY_BUCKETS", "20"),
        help="Comma-separated LAZY bucket sizes in seconds (default: 20)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "NPU"),
        choices=["CPU", "GPU", "NPU"],
        help="OpenVINO device for both encoder and decoder (default: NPU)",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG-level logging")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _LOGGER.info("wyoming-parakeet-on-intel-npu v%s", __version__)
    available = ov.Core().available_devices
    _LOGGER.info("OpenVINO devices available: %s", available)
    if args.device not in available:
        _LOGGER.error(
            "Requested device %s not available (have %s). "
            "Make sure the NPU device is passed through to the container.",
            args.device, available,
        )
        sys.exit(1)

    model_dir = os.path.join(args.data_dir, MODEL_NAME)
    encoder_onnx = os.path.join(model_dir, "encoder-model.onnx")
    decoder_ir = os.path.join(args.data_dir, "static_decoder", "decoder-static.xml")
    cache_dir = os.path.join(args.data_dir, "ov_cache")

    for required, label in [(encoder_onnx, "FP32 encoder ONNX"),
                            (decoder_ir, "static decoder IR")]:
        if not os.path.exists(required):
            _LOGGER.error(
                "%s not found at %s. The entrypoint should fetch it on first run; "
                "if you bypassed the entrypoint, run scripts/bootstrap.py manually.",
                label, required,
            )
            sys.exit(1)

    # 1. Load the onnx_asr pipeline (uses INT8 ORT sessions internally — we'll
    #    immediately replace them).
    _LOGGER.info("Loading onnx_asr pipeline (model=%s, quantization=int8) ...", MODEL_NAME)
    model = onnx_asr.load_model(
        model=MODEL_NAME,
        path=model_dir,
        providers=["CPUExecutionProvider"],
        sess_options=onnxruntime.SessionOptions(),
        quantization="int8",
    )

    # 2. Replace the encoder + decoder with OpenVINO shims.
    #    IMPORTANT: assign on `model.asr.*`, not on `model.*` — the adapter
    #    wrapper does not proxy attribute writes.
    eager = _parse_buckets(args.encoder_buckets)
    lazy = _parse_buckets(args.encoder_lazy_buckets)
    _LOGGER.info("Encoder buckets: eager=%s lazy=%s on %s", eager, lazy, args.device)
    model.asr._encoder = OpenVinoEncoderShim(
        onnx_path=encoder_onnx,
        device=args.device,
        cache_dir=cache_dir,
        eager_seconds=eager,
        lazy_seconds=lazy,
    )
    model.asr._decoder_joint = OpenVinoDecoderShim(
        ir_path=decoder_ir,
        device=args.device,
        cache_dir=cache_dir,
    )

    # 3. Run the wyoming server.
    server = AsyncServer.from_uri(args.uri)
    model_lock = asyncio.Lock()
    info = _build_wyoming_info(args.language)
    _LOGGER.info("Ready. Listening on %s", args.uri)
    await server.run(partial(
        ParakeetEventHandler, info, model, model_lock,
        default_language=args.language,
    ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
