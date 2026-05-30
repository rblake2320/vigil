# Spark1 RPC Worker Setup

Enables distributed inference of Step-3.7-Flash across Spark1+Spark2 via QSFP 200Gbps.

## What's installed
- Binary: `~/llama.cpp/build-rpc/bin/rpc-server-stepfun` (StepFun fork, protocol-matched)
- Libs: `~/llama.cpp/build-rpc/bin/libggml*.so.0` (copied from Spark2 StepFun build)
- Service: `~/.config/systemd/user/llama-rpc.service` (enabled, auto-restart)
- Interface: `192.168.100.10:50052` (QSFP only — never expose to open network)

## Management
```bash
# Status
ssh spark1 systemctl --user status llama-rpc.service

# Restart
ssh spark1 systemctl --user restart llama-rpc.service

# Logs
ssh spark1 journalctl --user -u llama-rpc.service -n 20
```

## How run_step37.sh uses it
Auto-detects via `nc -z -w 3 192.168.100.10 50052`.
- Reachable → `--rpc 192.168.100.10:50052 --n-gpu-layers 99` (all layers distributed)
- Not reachable → `--n-gpu-layers 18` (local-only fallback)

## Protocol note
MUST use StepFun fork binary on Spark1. Upstream llama.cpp rpc-server has
incompatible RPC protocol — will crash immediately with "malformed response".
