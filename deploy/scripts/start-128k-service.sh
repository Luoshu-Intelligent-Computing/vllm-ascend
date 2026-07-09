#!/bin/bash
# 310P vllm-ascend 128K 上下文服务启动脚本
# 镜像: registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708
# 硬件: Atlas 300I Duo (310P3 × 2)
# 验证日期: 2026-07-08

set -e

# ─────────────────────────────────────────
# 配置（按需修改）
# ─────────────────────────────────────────
IMAGE="registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708"
CONTAINER_NAME="vllm-310p-128k"
MODEL_PATH="/models/llm-service/vllm-ascend/Qwen3.6-35B-A3B-w8a8"
MODELS_VOLUME="/srv/meetai/models"
PORT=18082
SERVED_MODEL_NAME="qwen3.6-128k"

# ─────────────────────────────────────────
# 前置检查
# ─────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] 检查环境..."

# 检查镜像
if ! sudo podman image exists "$IMAGE" 2>/dev/null; then
    echo "[ERROR] 镜像不存在: $IMAGE"
    echo "  请先构建镜像，参考: Dockerfile.310p"
    exit 1
fi

# 停止并清理旧容器
if sudo podman container exists "$CONTAINER_NAME" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] 停止旧容器 $CONTAINER_NAME..."
    sudo podman stop "$CONTAINER_NAME" --time 15 2>/dev/null || true
    sudo podman rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# 检查 NPU 设备
if [ ! -e /dev/davinci0 ] || [ ! -e /dev/davinci1 ]; then
    echo "[ERROR] NPU 设备 /dev/davinci0 /dev/davinci1 不存在"
    exit 1
fi

# 检查模型路径
if [ ! -d "$MODELS_VOLUME" ]; then
    echo "[ERROR] 模型目录不存在: $MODELS_VOLUME"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] 启动服务..."

# ─────────────────────────────────────────
# 启动容器
# ─────────────────────────────────────────
sudo podman run -d \
    --name "$CONTAINER_NAME" \
    --privileged \
    --network host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v "$MODELS_VOLUME":/models:ro \
    "$IMAGE" \
    bash -c "
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        export ASCEND_CUSTOM_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom

        python3 -m vllm.entrypoints.openai.api_server \\
            --model $MODEL_PATH \\
            --served-model-name $SERVED_MODEL_NAME \\
            --host 0.0.0.0 \\
            --port $PORT \\
            -tp 2 \\
            --max-model-len 131072 \\
            --max-num-seqs 1 \\
            --max-num-batched-tokens 1024 \\
            --gpu-memory-utilization 0.75 \\
            --dtype float16 \\
            --kv-cache-dtype auto \\
            --trust-remote-code \\
            --enable-chunked-prefill \\
            --no-enable-prefix-caching \\
            --reasoning-parser qwen3 \\
            --additional-config '{\"ascend_compilation_config\": {\"fuse_norm_quant\": false}}' \\
            --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [1]}' \\
            --async-scheduling \\
            --mamba-ssm-cache-dtype float16 \\
            --allowed-local-media-path /
    "

echo "[$(date '+%H:%M:%S')] 容器已启动，等待服务就绪..."

# ─────────────────────────────────────────
# 等待服务就绪（最多 10 分钟）
# ─────────────────────────────────────────
TIMEOUT=600
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s "http://localhost:$PORT/health" -o /dev/null 2>/dev/null; then
        echo ""
        echo "[$(date '+%H:%M:%S')] ✅ 服务已就绪！"
        echo ""
        echo "  API 端点: http://localhost:$PORT"
        echo "  模型名:   $SERVED_MODEL_NAME"
        echo "  最大上下文: 131072 tokens (128K)"
        echo ""
        echo "  快速测试:"
        echo "    curl http://localhost:$PORT/v1/chat/completions \\"
        echo "      -H 'Content-Type: application/json' \\"
        echo "      -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":200}'"
        echo ""
        exit 0
    fi

    log=$(sudo podman logs --tail 1 "$CONTAINER_NAME" 2>/dev/null | tail -1 | cut -c1-100)
    printf "\r[%ds] %s          " $ELAPSED "$log"
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

echo ""
echo "[WARN] 超时 ${TIMEOUT}s，服务可能仍在启动中"
echo "  查看日志: sudo podman logs -f $CONTAINER_NAME"
