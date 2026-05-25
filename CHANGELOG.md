# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Initial preview

### Added
- Wyoming protocol STT server bundling Parakeet TDT 0.6B v3 multilingual on Intel NPU via OpenVINO
- Multi-bucket encoder dispatch (eager + lazy buckets) to handle variable-length audio on a static-shape NPU
- Decoder/joint compiled once with static per-token shapes and run on NPU
- First-run bootstrap: downloads model weights from HuggingFace, builds static decoder IR
- Dockerfile based on Ubuntu 24.04 with bundled Level Zero loader 1.28.6 and Intel NPU driver 1.32.1
- `LANGUAGE`, `DEVICE`, `ENCODER_BUCKETS`, `ENCODER_LAZY_BUCKETS` env-var configuration
- Health check on the Wyoming TCP port
- `docker-compose.example.yml` for drop-in HA integration
- Documentation: README, HARDWARE.md, TROUBLESHOOTING.md
