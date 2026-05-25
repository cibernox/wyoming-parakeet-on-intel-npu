# Troubleshooting

## "OpenVINO devices available: ['CPU']" — NPU not detected

The container can't see the NPU. Check, in order:

1. **Is the device node passed through?**
   ```bash
   docker exec wyoming-parakeet-npu ls -l /dev/accel/accel0
   ```
   If not present, your `docker run` is missing `--device /dev/accel/accel0` (or compose's `devices:` block).

2. **Is the host kernel module loaded?**
   ```bash
   lsmod | grep intel_vpu
   dmesg | grep intel_vpu
   ```
   If not, your kernel is too old or `intel_vpu` was blacklisted.

3. **LXC layer (Proxmox / containerd-shim users)?**
   See [docs/HARDWARE.md → Proxmox LXC passthrough](HARDWARE.md#proxmox-lxc-passthrough). The device has to be passed through to the LXC before Docker inside the LXC can see it.

4. **Inside the container, do you see the userspace driver?**
   ```bash
   docker exec wyoming-parakeet-npu ls /usr/lib/x86_64-linux-gnu/libze_intel_npu*
   ```
   Should show `libze_intel_npu.so.1` and family. If missing, the image build went wrong.

## OOM-kill during NPU compile

Symptoms: container suddenly exits, `dmesg | grep oom-kill` on the host shows the python process being killed.

Causes:
- Container's `mem_limit` is too low (default 8 GB; cold compile of large buckets can need ~10 GB)
- Parent LXC has hit its limit and OOM-killed everything inside
- Other containers on the host are competing for memory

Fixes:
- Bump compose `mem_limit` to `12g`
- Reduce the bucket count (e.g., remove `60`-sec bucket if you don't need it)
- Make the lazy bucket list shorter — lazy buckets are placeholders at startup, so they cost nothing until first hit

## "audio N frames > largest bucket M — truncating" warning

Your audio is longer than the largest configured bucket. The encoder runs on the largest bucket and audio beyond that is truncated.

Fix: add a larger bucket. For dictation use cases:

```yaml
environment:
  ENCODER_BUCKETS: "10,30"
  ENCODER_LAZY_BUCKETS: "60"
```

## First-boot is slow (~60-90 s)

Expected. Cold first run does:

- Download ~3.2 GB of model files from HuggingFace
- Compile each eager NPU bucket (~12 s each)
- Build the static decoder IR (~3 s)

Subsequent restarts hit `/data/ov_cache/` and start in ~3-5 s. If you're seeing repeated slow boots, your `/data` volume isn't persistent.

## Transcripts are correct but slower than expected

Possible causes:
- `DEVICE=CPU` set instead of `DEVICE=NPU` (check the startup log)
- Audio is longer than your eager bucket and triggering lazy compile each time. After the first lazy compile, the bucket is hot for the remainder of the container's lifetime — but if the container is restarting frequently, you'll pay the lazy compile each restart. Bump it to eager:
  ```yaml
  ENCODER_BUCKETS: "5,20"
  ENCODER_LAZY_BUCKETS: ""
  ```

## "Output names mismatch between OpenVINO and ONNX"

If you see this in logs, you're hitting a known incompatibility between `onnxruntime-openvino` and INT8 Parakeet ONNX. **This image doesn't use that path** — it talks to OpenVINO directly. If you see this error, you're somehow running a different code path than the image entrypoint. Check that you're using `ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]` (the default).

## NPU compile fails: "Upper bounds are not specified for node ..."

You're trying to feed a fully-dynamic ONNX to NPU compile. The image avoids this by reshaping every encoder bucket to a static `T` before compile, so this shouldn't happen. If you see it, file a bug — most likely a code path that bypasses our static-shape preprocessing.

## Home Assistant doesn't pick up the server

Standard Wyoming protocol checklist:

- Port 10300 is reachable from HA (`telnet <host> 10300` from HA's container should connect)
- HA's Wyoming integration is configured with `tcp://<host>:10300`
- Container logs show `Ready. Listening on tcp://0.0.0.0:10300` and isn't crashing on requests
- HA assist pipeline has Speech-to-Text set to this Wyoming source

## Filing a bug

When opening an issue, please attach:

1. Your CPU model (`cat /proc/cpuinfo | grep 'model name' | head -1`)
2. Kernel version (`uname -r`)
3. Output of `ls -l /dev/accel/accel0` (or "doesn't exist")
4. Container startup log (`docker logs wyoming-parakeet-npu 2>&1 | head -50`)
5. The actual error/symptom you're seeing
