#!/bin/bash
# Step-3.7-Flash server launcher
# Model: IQ4_XS, 105GB, 256K context, port 8898
# Vision: mmproj-step3.7-flash-f16.gguf (3.97GB) — enables image input
# RPC: distributes layers to Spark1 via QSFP 200Gbps (192.168.100.10:50052)

MODEL_DIR=~/models/gguf/step-3.7-flash
BIN=~/stepfun-llama/build-cuda/bin/llama-server
MMPROJ="$MODEL_DIR/mmproj-step3.7-flash-f16.gguf"
RPC_HOST="192.168.100.10:50052"   # Spark1 QSFP interface
RPC_PING_TIMEOUT=3                 # seconds to wait for rpc-server

if [ ! -f "$BIN" ]; then
    echo "ERROR: llama-server not built. Run: cd ~/stepfun-llama/build-cuda && cmake --build build-cuda --target llama-server rpc-server -j\$(nproc)"
    exit 1
fi

SHARD1=$(find "$MODEL_DIR" -name "*IQ4_XS*00001*" 2>/dev/null | head -1)
if [ -z "$SHARD1" ]; then
    echo "ERROR: Model shards not found in $MODEL_DIR"
    exit 1
fi

echo "Starting Step-3.7-Flash on port 8898..."
echo "Model: $SHARD1"

# Vision projector
VISION_ARGS=""
if [ -f "$MMPROJ" ]; then
    echo "Vision: $MMPROJ (enabled)"
    VISION_ARGS="--mmproj $MMPROJ --chat-template chatml"
else
    echo "Vision: mmproj not found — text-only mode"
    echo "  Download: huggingface-hub download stepfun-ai/Step-3.7-Flash-GGUF mmproj-step3.7-flash-f16.gguf --local-dir $MODEL_DIR"
fi

# RPC distributed inference (Spark1 via QSFP 200Gbps)
RPC_ARGS=""
GPU_LAYERS=18   # fallback: local only
if nc -z -w "$RPC_PING_TIMEOUT" "${RPC_HOST%%:*}" "${RPC_HOST##*:}" 2>/dev/null; then
    echo "RPC: Spark1 reachable at $RPC_HOST — distributing layers across both Sparks"
    RPC_ARGS="--rpc $RPC_HOST"
    GPU_LAYERS=99   # push all layers; RPC worker handles the overflow
else
    echo "RPC: Spark1 not reachable ($RPC_HOST) — local-only mode ($GPU_LAYERS GPU layers)"
    echo "  Start Spark1 worker: ssh spark1 'nohup ~/llama.cpp/build-rpc/bin/rpc-server --host 192.168.100.10 --port 50052 > /tmp/rpc-server.log 2>&1 &'"
fi

LD_LIBRARY_PATH=~/stepfun-llama/build-cuda/bin:$LD_LIBRARY_PATH \
"$BIN" \
    --model "$SHARD1" \
    --host 0.0.0.0 \
    --port 8898 \
    --ctx-size 16384 \
    --n-gpu-layers $GPU_LAYERS \
    --threads 8 \
    --parallel 1 \
    --flash-attn on \
    --log-prefix \
    -fit off \
    $VISION_ARGS \
    $RPC_ARGS \
    "$@"
