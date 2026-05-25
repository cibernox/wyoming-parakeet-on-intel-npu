# Hardware requirements & setup

## Supported CPUs

Intel Core Ultra processors with the AI Boost NPU.

| Family | Codename | NPU generation | Tested? |
|---|---|---|---|
| Core Ultra (Series 1) | Meteor Lake | NPU 3720 | Should work |
| Core Ultra (Series 2) | Arrow Lake | NPU 3720 (same IP as MTL) | ✅ Tested |
| Core Ultra (Series 2) | Lunar Lake | NPU 4 | Should work |
| Core Ultra (Series 3) | Panther Lake | NPU 5 | Untested |

If the NPU PCI ID isn't recognized by your kernel's `intel_vpu` driver, the device won't appear and this image won't work.

## Verify NPU presence on the host

Before deploying:

```bash
# 1. Device node should exist
ls -l /dev/accel/accel0
# crw-rw---- 1 root render 261, 0 ...  <- good

# 2. Driver should be loaded
lsmod | grep intel_vpu

# 3. PCI device should be Intel NPU
lspci -nn -k -d 8086: | grep -A2 -i "npu\|ai boost"
```

If `/dev/accel/accel0` doesn't exist, your kernel may be too old (need ≥ 6.10) or `intel_vpu` may not be loaded. On Debian/Ubuntu:

```bash
sudo modprobe intel_vpu
echo intel_vpu | sudo tee /etc/modules-load.d/intel-vpu.conf
```

## Docker passthrough

```bash
docker run --device /dev/accel/accel0:/dev/accel/accel0 ...
```

Or in compose:

```yaml
devices:
  - /dev/accel/accel0:/dev/accel/accel0
```

The container needs access to the device node. The image runs as root inside the container, so group permissions on the host don't matter — but the device must exist and have major:minor `261:0`.

## Proxmox LXC passthrough

If you're running Docker inside a Proxmox LXC container (a common self-hosted layout), the LXC needs the NPU device passed through first.

Edit `/etc/pve/lxc/<vmid>.conf`:

```
dev2: /dev/accel/accel0,gid=992
```

Where `gid=992` should match the `render` group inside the LXC (`getent group render` to check). Then reboot the LXC:

```bash
pct reboot <vmid>
```

After this, both `/dev/accel/accel0` should be visible inside the LXC and Docker can passthrough as normal.

## Memory

The container peaks at ~5 GB during cold compile of the 20 s bucket. Default `mem_limit: 8g` is comfortable; reduce to 6 GB at minimum if memory is tight.

If you've configured very large buckets (e.g., `60`), give it more — a 60 s bucket compile peaks at ~10 GB.

## Storage

The `/data` volume holds:

- ~3.2 GB of model weights (downloaded from HuggingFace on first run)
- ~150-300 MB per compiled NPU bucket blob in `/data/ov_cache/`
- HuggingFace cache (small)

Plan for ~5 GB on the volume.

## Falling back to CPU/GPU

If you don't have an NPU but want to try the image anyway, set `DEVICE=CPU` (or `GPU` for the iGPU). The same OpenVINO pipeline works on those devices — performance will be lower than NPU but still works. The image will refuse to start if the requested device isn't detected.
