#!/bin/bash
# 香橙派 Atlas 200I Pro（310P1 SoC，单卡）部署脚本
# 镜像: 310p-opt-openeuler-20260709（含 GDN AscendC kernel，源码烘焙）
# 与 ails-a1 start-128k-service.sh 的差异：
#   - tp=1（单卡）；无 davinci1 / devmm_svm
#   - npu-smi 在 /usr/local/sbin/（非 /usr/local/bin/）
#   - ascend-toolkit 不在宿主机，由容器内部 set_env.sh 处理
#   - dcmi 宿主机路径是文件而非目录，不挂载
#   - libtensorflow/aicpu_kernels 来自 /usr/lib64/
set -e
IMAGE="registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-openeuler-20260709"
NAME="qwen36-35b-310p-tp1-poc"

docker stop "$NAME" 2>/dev/null || true
docker rm -f "$NAME" 2>/dev/null || true

docker run -d \
  --name "$NAME" \
  --privileged \
  --network host \
  --shm-size=64g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e ASCEND_VISIBLE_DEVICES=0 \
  -e OMP_PROC_BIND=false \
  -e OMP_NUM_THREADS=1 \
  -e ASCEND_GLOBAL_LOG_LEVEL=3 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:False \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  -e HCCL_OP_EXPANSION_MODE=AIV \
  -e PYTHONUNBUFFERED=1 \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/lib64/libtensorflow.so:/usr/lib64/libtensorflow.so:ro \
  -v /usr/lib64/aicpu_kernels:/usr/lib64/aicpu_kernels:ro \
  -v /models:/models \
  "$IMAGE" \
  bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_CUSTOM_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom

    python3 -m vllm.entrypoints.openai.api_server \
      --model /models/Qwen3.6-35B-A3B-w8a8 \
      --served-model-name qwen3.6 \
      --host 0.0.0.0 \
      --port 38081 \
      -tp 1 \
      --max-model-len 65536 \
      --max-num-seqs 1 \
      --max-num-batched-tokens 2048 \
      --gpu-memory-utilization 0.6 \
      --dtype float16 \
      --kv-cache-dtype auto \
      --trust-remote-code \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --reasoning-parser qwen3 \
      --compilation-config '"'"'{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'"'"' \
      --additional-config '"'"'{"ascend_compilation_config": {"fuse_norm_quant": false}}'"'"' \
      --mamba-ssm-cache-dtype float16 \
      --allowed-local-media-path /
  '

echo "Started (310p-opt-openeuler, tp=1)"
docker ps --filter "name=$NAME"
