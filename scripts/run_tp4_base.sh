#!/usr/bin/env bash
# Launch an isolated TP=4 RadixArk/DFlash2 profile. K=12 is the reference.
# Run this on the Ray head. It starts only containers named experimental-radixark-*.
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/timothystewart6/vllm-gb10@sha256:60994dfa59451aabde6091d0391ac89fc7843a6a66f35c9d9535ad27477357f2}"
HEAD_IP="${HEAD_IP:?Set HEAD_IP to the Ray head address}"
WORKER_IPS="${WORKER_IPS:?Set WORKER_IPS to three space-separated worker addresses}"
read -r -a WORKERS <<< "$WORKER_IPS"
SSH_USER="${SSH_USER:?Set SSH_USER for worker SSH access}"
HF_CACHE="${HF_CACHE:?Set HF_CACHE to the shared Hugging Face cache path}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH to the target checkpoint}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH to the DFlash2 checkpoint}"
QSFP_IFACE="${QSFP_IFACE:?Set QSFP_IFACE to the RoCE interface}"
IB_HCA="${IB_HCA:?Set IB_HCA to the RoCE HCA}"
GID_INDEX="${NCCL_IB_GID_INDEX:?Set NCCL_IB_GID_INDEX to the RoCE GID index}"
SHM_SIZE="${SHM_SIZE:-32g}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-4294967296}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dflash}"
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-12}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
# Optional native vLLM scheduler policy. Example: [[1,1,12],[2,4,5],[5,64,0]].
# K stays bounded by SPECULATIVE_TOKENS and is selected per global scheduler batch.
DYNAMIC_SD_SCHEDULE="${DYNAMIC_SD_SCHEDULE:-}"
DFLASH2_TRACE_ROOT="${DFLASH2_TRACE_ROOT:-}"
if ! [[ "$SPECULATIVE_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPECULATIVE_TOKENS must be a positive integer" >&2
  exit 2
fi
if [[ "$SPECULATIVE_METHOD" != "dflash" && "$SPECULATIVE_METHOD" != "mtp" ]]; then
  echo "SPECULATIVE_METHOD must be dflash or mtp" >&2
  exit 2
fi
if ! [[ "$MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_NUM_SEQS must be a positive integer" >&2
  exit 2
fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" && ! "$MAX_NUM_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_NUM_BATCHED_TOKENS must be empty or a positive integer" >&2
  exit 2
fi
dynamic_config_suffix=""
if [[ -n "$DYNAMIC_SD_SCHEDULE" ]]; then
  dynamic_config_suffix=",\"num_speculative_tokens_per_batch_size\":$DYNAMIC_SD_SCHEDULE"
fi
if [[ "$SPECULATIVE_METHOD" == "dflash" ]]; then
  speculative_config_json="{\"method\":\"dflash\",\"model\":\"/models/draft\",\"num_speculative_tokens\":$SPECULATIVE_TOKENS,\"draft_tensor_parallel_size\":1$dynamic_config_suffix}"
else
  if [[ -n "$DYNAMIC_SD_SCHEDULE" ]]; then
    echo "DYNAMIC_SD_SCHEDULE is currently validated only with DFlash2" >&2
    exit 2
  fi
  speculative_config_json="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPECULATIVE_TOKENS}"
fi
HEAD_NAME="experimental-radixark-${SPECULATIVE_METHOD}-tp4-k${SPECULATIVE_TOKENS}"
# Compiled graph shapes include K. Never share the K=12 cache with another K.
CACHE_ROOT="${CACHE_ROOT:-/var/tmp/vllm-experimental-${SPECULATIVE_METHOD}-k${SPECULATIVE_TOKENS}-tp4}"

common_args=(
  --network host --ipc host --shm-size "$SHM_SIZE" --gpus all
  --privileged --device /dev/infiniband:/dev/infiniband --ulimit memlock=-1:-1
  -e UCX_NET_DEVICES="$QSFP_IFACE"
  -e NCCL_SOCKET_IFNAME="$QSFP_IFACE"
  -e OMPI_MCA_btl_tcp_if_include="$QSFP_IFACE"
  -e GLOO_SOCKET_IFNAME="$QSFP_IFACE"
  -e TP_SOCKET_IFNAME="$QSFP_IFACE"
  -e NCCL_IB_DISABLE=0 -e NCCL_NET_GDR_LEVEL=0
  -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$GID_INDEX"
  -e NCCL_ALGO=Ring -e NCCL_MIN_NCHANNELS=4
  -e RAY_memory_monitor_refresh_ms=0 -e RAY_num_prestart_python_workers=0
  -e RAY_object_store_memory="$RAY_OBJECT_STORE_MEMORY"
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  -e SAFETENSORS_FAST_GPU=1 -e VLLM_ATTENTION_BACKEND=FLASHINFER
  -e OMP_NUM_THREADS=4
  -v "$HF_CACHE:/root/.cache/huggingface"
  -v "$TARGET_MODEL_PATH:/models/target:ro"
  -v "$DRAFT_MODEL_PATH:/models/draft:ro"
)
if [[ -n "$DFLASH2_TRACE_ROOT" ]]; then
  common_args+=(
    -e VLLM_DFLASH2_TRACE_PATH="/traces/${HEAD_IP}.jsonl"
    -v "$DFLASH2_TRACE_ROOT:/traces"
  )
fi

for worker in "${WORKERS[@]}"; do
  node_id="${worker##*.}"
  ssh -o BatchMode=yes "$SSH_USER@$worker" \
    "mkdir -p '$CACHE_ROOT' '$DFLASH2_TRACE_ROOT' && docker rm -f experimental-radixark-dflash-tp4-worker-$node_id >/dev/null 2>&1 || true"
done
docker ps -a --format '{{.Names}}' | grep -E '^experimental-radixark-(dflash|mtp)-tp4-k[0-9]+$' \
  | xargs -r docker rm -f >/dev/null 2>&1 || true
# Port 8000 is owned by this experiment's current single-node instance. Preserve
# its container so it can be restarted after the TP=4 measurement.
docker stop experimental-dflash-single-node >/dev/null 2>&1 || true

# Head starts Ray first and waits for all four nodes before vLLM begins loading.
mkdir -p "$CACHE_ROOT"
docker run -d --name "$HEAD_NAME" \
  "${common_args[@]}" \
  -e VLLM_HOST_IP="$HEAD_IP" -e MASTER_ADDR="$HEAD_IP" -e RAY_ADDRESS="$HEAD_IP:6379" \
  -v "$CACHE_ROOT:/root/.cache/vllm" \
  --entrypoint /bin/bash "$IMAGE" -lc "
    ray stop -f >/dev/null 2>&1 || true;
    ray start --head --node-ip-address=$HEAD_IP --port=6379 --temp-dir /tmp/ray-tp4-k$SPECULATIVE_TOKENS \
      --object-store-memory $RAY_OBJECT_STORE_MEMORY --num-cpus 2 \
      --include-dashboard=false --disable-usage-stats &&
    alive=0;
    for i in \$(seq 1 150); do
      alive=\$(python3 -c 'import ray; ray.init(address=\"auto\"); print(sum(n[\"Alive\"] for n in ray.nodes()))');
      [ \"\$alive\" -ge 4 ] && break;
      sleep 2;
    done;
    [ \"\$alive\" -ge 4 ] || { echo \"Timed out waiting for four Ray nodes; found \$alive\"; exit 1; };
    vllm serve /models/target --host 0.0.0.0 --port 8000 \
      --served-model-name "$HEAD_NAME" \
      --tensor-parallel-size 4 --distributed-executor-backend ray \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --load-format fastsafetensors \
      --kv-cache-dtype fp8_e4m3 --max-model-len 65536 --max-num-seqs "$MAX_NUM_SEQS" \
      ${MAX_NUM_BATCHED_TOKENS:+--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"} \
      --enable-prefix-caching --reasoning-parser qwen3 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --speculative-config '$speculative_config_json'
  "

for worker in "${WORKERS[@]}"; do
  node_id="${worker##*.}"
  ssh -o BatchMode=yes "$SSH_USER@$worker" "
    mkdir -p '$CACHE_ROOT' &&
    docker run -d --name experimental-radixark-dflash-tp4-worker-$node_id \
      --network host --ipc host --shm-size '$SHM_SIZE' --gpus all \
      --privileged --device /dev/infiniband:/dev/infiniband --ulimit memlock=-1:-1 \
      -e VLLM_HOST_IP='$worker' -e MASTER_ADDR='$HEAD_IP' \
      -e UCX_NET_DEVICES='$QSFP_IFACE' -e NCCL_SOCKET_IFNAME='$QSFP_IFACE' \
      -e OMPI_MCA_btl_tcp_if_include='$QSFP_IFACE' -e GLOO_SOCKET_IFNAME='$QSFP_IFACE' \
      -e TP_SOCKET_IFNAME='$QSFP_IFACE' -e NCCL_IB_DISABLE=0 -e NCCL_NET_GDR_LEVEL=0 \
      -e NCCL_IB_HCA='$IB_HCA' -e NCCL_IB_GID_INDEX='$GID_INDEX' -e NCCL_ALGO=Ring \
      -e NCCL_MIN_NCHANNELS=4 -e RAY_memory_monitor_refresh_ms=0 \
      -e RAY_num_prestart_python_workers=0 -e RAY_object_store_memory='$RAY_OBJECT_STORE_MEMORY' \
      -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 -e SAFETENSORS_FAST_GPU=1 \
      -e VLLM_ATTENTION_BACKEND=FLASHINFER -e OMP_NUM_THREADS=4 \
      ${DFLASH2_TRACE_ROOT:+-e VLLM_DFLASH2_TRACE_PATH=/traces/$worker.jsonl -v "$DFLASH2_TRACE_ROOT:/traces"} \
      -v '$HF_CACHE:/root/.cache/huggingface' \
      -v '$TARGET_MODEL_PATH:/models/target:ro' \
      -v '$DRAFT_MODEL_PATH:/models/draft:ro' \
      -v '$CACHE_ROOT:/root/.cache/vllm' --entrypoint /bin/bash '$IMAGE' -lc \
      'until ray start --address=$HEAD_IP:6379 --node-ip-address=$worker --object-store-memory=$RAY_OBJECT_STORE_MEMORY --num-cpus 2 --disable-usage-stats --block; do sleep 2; done'
  "
done

for i in $(seq 1 240); do
  curl -fsS http://127.0.0.1:8000/health >/dev/null && {
    echo "$HEAD_NAME is healthy"
    exit 0
  }
  docker inspect -f '{{.State.Running}}' "$HEAD_NAME" 2>/dev/null | grep -qx true || {
    docker logs --tail 80 "$HEAD_NAME"
    exit 1
  }
  sleep 5
done

echo "$HEAD_NAME did not become healthy within 20 minutes" >&2
docker logs --tail 80 "$HEAD_NAME" >&2
exit 1
