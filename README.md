# wyoming-parakeet-on-intel-npu

Run NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) multilingual speech-to-text model on the Intel NPU ("AI Boost") shipped with Core Ultra processors, exposed over the [Wyoming protocol](https://github.com/rhasspy/wyoming) for direct integration with Home Assistant.

**3-6× faster than running on CPU and 10-22× more energy-efficient per inference**, on identical hardware. ([Detailed benchmark results.](#performance))

## What this is (and isn't)

This image is **opinionated and narrowly scoped on purpose**:

- ✅ Parakeet TDT 0.6B v3 multilingual (25 European languages)
- ✅ Intel NPU (Meteor Lake / Arrow Lake / Lunar Lake)
- ✅ Wyoming protocol → drop-in for Home Assistant's "Speech-to-Phrase" / "Assist" stack
- ✅ Falls back to OpenVINO CPU/GPU if NPU isn't available

- ❌ No other ASR models (no Whisper, no Canary, no Gigaam)
- ❌ No AMD Ryzen AI / Apple Neural Engine / Qualcomm NPU support — different driver stacks entirely
- ❌ No streaming/incremental ASR (the upstream wyoming-onnx-asr architecture batches the full utterance and runs once at AudioStop)

If you need any of the ❌ items, look at [tboby/wyoming-onnx-asr](https://github.com/tboby/wyoming-onnx-asr) for the broader, less specialized version this project was forked from.

## Quick start

### Hardware requirements

- An Intel Core Ultra processor with the AI Boost NPU
  - Verified hardware: **Core Ultra 7 265K** (Arrow Lake)
  - Should also work on Meteor Lake (Core Ultra 7 155H/165H, etc.) and Lunar Lake (Core Ultra 7 258V, etc.)
- Linux kernel with `intel_vpu` driver (kernel ≥ 6.10 ships it)
- `/dev/accel/accel0` must exist and be readable by the container's UID

Verify before starting:

```bash
ls /dev/accel/accel0     # should show a char device
lspci | grep -i npu      # should show "Processing accelerators ... Intel ... NPU"
```

### Run with Docker

```bash
docker run -d \
  --name wyoming-parakeet-npu \
  --device /dev/accel/accel0 \
  -e LANGUAGE=es \
  -p 10300:10300 \
  -v parakeet-data:/data \
  --restart unless-stopped \
  ghcr.io/cibernox/wyoming-parakeet-on-intel-npu:latest
```

First boot downloads ~3.2 GB of model files from HuggingFace and compiles the 5 s NPU bucket — ~60-90 seconds total. Subsequent restarts are < 5 seconds because the compiled blobs are cached in the volume.

### Run with Docker Compose

See [`docker-compose.example.yml`](./docker-compose.example.yml).

### Use from Home Assistant

In the Wyoming integration, point the Speech-to-Text endpoint at `tcp://<host>:10300`. The default language served by the model is set via `LANGUAGE=` (default `en`); per-request languages from HA override the default automatically.

## Configuration

All configuration is via env vars (or matching CLI flags if you prefer to pass them directly).

| Env var | Default | Description |
|---|---|---|
| `LANGUAGE` | `en` | Default transcription language. Used when the client request doesn't specify one. Must be one of: `en, es, fr, de, it, pt, nl, pl, ru, uk, bg, hr, cs, da, et, fi, el, hu, lv, lt, mt, ro, sk, sl, sv` |
| `DEVICE` | `NPU` | OpenVINO device to use for both encoder and decoder. `NPU`, `GPU`, or `CPU`. The image will refuse to start if the requested device isn't visible. |
| `ENCODER_BUCKETS` | `5` | Comma-separated **eager** bucket sizes in seconds. Compiled at startup, kept in memory. |
| `ENCODER_LAZY_BUCKETS` | `20` | Comma-separated **lazy** bucket sizes in seconds. Compiled on first request that needs them, then kept hot. |
| `WYOMING_URI` | `tcp://0.0.0.0:10300` | Wyoming server URI |
| `DATA_DIR` | `/data` | Where model files and the OpenVINO compile cache live. Must be a persistent volume. |

### What's a "bucket" and how do I pick the right ones?

The Intel NPU compiler requires static input shapes — it bakes the encoder's tile sizes and memory layout into the compiled blob based on the audio length. We sidestep that by pre-compiling several static-shape encoders ("buckets") and routing each request to the smallest one that fits.

For typical Home Assistant smart-home commands the defaults (`5` eager + `20` lazy) work well: most utterances are < 5 s and hit the eager bucket; longer dictation lazily compiles the 20 s bucket on first need (one-time ~12 s cold compile, then cached). Audio longer than the largest bucket is truncated.

If you mostly transcribe longer dictation, set:

```yaml
environment:
  ENCODER_BUCKETS: "10,30"
  ENCODER_LAZY_BUCKETS: "60"
```

## Performance

Measured on a Core Ultra 7 265K (Arrow Lake) running the same Spanish audio through both the plain ORT-CPU + INT8 baseline (the upstream wyoming-onnx-asr-cpu image) and this NPU-targeted image:

| Audio | Backend | Avg latency | Energy / inference | Power above idle |
|---|---|---|---|---|
| 10 s | CPU INT8 | 978 ms | 44.6 J | 45.6 W |
| 10 s | **NPU FP32** | **204 ms** | **4.2 J** | **20.5 W** |
| 20 s | CPU INT8 | 1 708 ms | 79.8 J | 46.7 W |
| 20 s | **NPU FP32** | **615 ms** | **7.8 J** | **12.7 W** |
| 60 s | CPU INT8 | 5 011 ms | 237.7 J | 47.4 W |
| 60 s | **NPU FP32** | **818 ms** | **11.0 J** | **13.4 W** |

The NPU draws about half the package power of the CPU during transcription, runs the model 3-6× faster, and uses an order of magnitude less energy per inference. The CPU is also mostly idle while the NPU works, so other workloads on the same machine (LLM inference on the iGPU, etc.) aren't competing for it.

## How does it work?

1. The container loads `onnx_asr` with the Parakeet INT8 ONNX (small, fast to construct).
2. It then **replaces** `onnx_asr`'s ORT encoder + decoder with OpenVINO NPU shims that compile the FP32 model to NPU-native blobs.
3. The encoder shim manages multiple compiled buckets (different static `T` sizes) and dispatches each request to the smallest fit. Lazy-loaded buckets compile on first hit (~12 s cold, ~1 s from `CACHE_DIR`).
4. The decoder runs on the NPU at fixed shapes (per-token), so a single compiled blob handles every utterance.
5. The Wyoming protocol layer is unchanged from upstream `wyoming-onnx-asr` — full HA compatibility.

## Building from source

```bash
git clone https://github.com/cibernox/wyoming-parakeet-on-intel-npu.git
cd wyoming-parakeet-on-intel-npu
docker build -t wyoming-parakeet-npu:dev .
```

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Most likely:

- **`OpenVINO devices available: ['CPU']`** — the NPU isn't being passed through to the container. Check `--device /dev/accel/accel0` is on the `docker run` command and that the host actually has an NPU (`ls /dev/accel/accel0`).
- **`Load Library of libze_intel_vpu.so.1 failed`** — old Level Zero loader on the host. Doesn't apply when running in this image (we ship loader 1.28.6) but does apply if you're running outside.
- **`audio N frames > largest bucket M — truncating`** — audio longer than your largest bucket. Increase `ENCODER_LAZY_BUCKETS` to add a bigger one.
- **OOM during NPU compile** — give the container more memory (`mem_limit: 12g`). Cold compile of large buckets peaks at ~6-10 GB.

## Acknowledgements & upstreams

- [tboby/wyoming-onnx-asr](https://github.com/tboby/wyoming-onnx-asr) — the original Wyoming server this is forked from. The protocol handler is essentially theirs (MIT).
- [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr) — the underlying ASR pipeline library; we monkey-patch its encoder/decoder slots with OpenVINO shims (MIT).
- [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) — the ONNX export of NVIDIA's Parakeet TDT 0.6B v3 model (CC-BY-4.0).
- [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) — Intel's inference runtime; the NPU plugin lives here (Apache-2.0).
- [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver) — Intel NPU userspace driver (MIT).
- [oneapi-src/level-zero](https://github.com/oneapi-src/level-zero) — Level Zero loader required to discover the NPU driver (MIT).

## License

[MIT](LICENSE)
