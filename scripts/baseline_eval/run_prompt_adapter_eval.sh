#!/usr/bin/env bash
set -euo pipefail

# 我们的方法：Prompt Embedding Adapter baseline。
# 默认跑 porn、gore 和 5 个 IP。每个数据集会先切 shard，然后按 GPU 并行生成。

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CONDA_ENV="${CONDA_ENV:-loraretrieval}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-4,5,6,7}}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-9}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.0}"
SAMPLE_SEED="${SAMPLE_SEED:-20260715}"
GENERATION_SEED="${GENERATION_SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baselines/prompt_adapter}"
LOG_DIR="${LOG_DIR:-outputs/logs/baseline_eval}"

PORN_CKPT="${PORN_CKPT:-outputs/safe_embedding_adapter/mlp_adapter/porn_mlp_depth2_ce/checkpoints/latest.pt}"
GORE_CKPT="${GORE_CKPT:-outputs/safe_embedding_adapter/mlp_adapter/gore_mlp_depth2_ce/checkpoints/latest.pt}"
IP_CKPT="${IP_CKPT:-outputs/safe_embedding_adapter/mlp_adapter/ip5_mlp_depth3_hinge/checkpoints/latest.pt}"

mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
GPU_COUNT="${#GPUS[@]}"
if [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "[adapter] GPU_IDS is empty" >&2
  exit 1
fi

SPLIT_NUM_PROMPTS_ARGS=()
if [[ -n "${NUM_PROMPTS:-}" ]]; then
  SPLIT_NUM_PROMPTS_ARGS=(--num_prompts "${NUM_PROMPTS}")
fi

EXTRA_ADAPTER_ARGS=()
if [[ "${RESTRICT_USER_CONTENT_TOKENS:-0}" == "1" ]]; then
  EXTRA_ADAPTER_ARGS=(--restrict_adapter_to_user_content_tokens)
fi

run_one() {
  local name="$1"
  local input_csv="$2"
  local target_risk="$3"
  local adapter_ckpt="$4"
  local output_dir="${OUTPUT_ROOT}/${name}_seed${GENERATION_SEED}"
  local shard_dir="${output_dir}/csv_shards"

  if [[ -d "${output_dir}/adapter" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[adapter] skip ${name}: ${output_dir}/adapter already exists; set OVERWRITE=1 to rerun"
    return
  fi

  echo "[adapter] split ${name}: ${input_csv}"
  conda run -n "${CONDA_ENV}" python scripts/baseline_eval/split_csv_round_robin.py \
    --input_csv "${input_csv}" \
    --output_dir "${shard_dir}" \
    --num_shards "${GPU_COUNT}" \
    --sample_seed "${SAMPLE_SEED}" \
    "${SPLIT_NUM_PROMPTS_ARGS[@]}"

  echo "[adapter] start ${name}: ${GPU_COUNT} GPU shard(s)"
  local pids=()
  for shard_index in "${!GPUS[@]}"; do
    local gpu="${GPUS[${shard_index}]}"
    local shard_csv="${shard_dir}/shard_$(printf '%03d' "${shard_index}").csv"
    local shard_log="${LOG_DIR}/prompt_adapter_${name}_gpu${gpu}_shard${shard_index}.log"
    local shard_rows
    shard_rows="$(($(wc -l < "${shard_csv}") - 1))"
    if [[ "${shard_rows}" -le 0 ]]; then
      echo "[adapter] skip empty shard ${name} shard=${shard_index}"
      continue
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" conda run -n "${CONDA_ENV}" python -u test_prompt_embedding_adapter_generation.py \
      --adapter_ckpt "${adapter_ckpt}" \
      --target_risk "${target_risk}" \
      --unsafe_csv "${shard_csv}" \
      --prompt_set unsafe \
      --use_all_data \
      --num_prompts "${shard_rows}" \
      --sample_seed "${SAMPLE_SEED}" \
      --generation_seed "${GENERATION_SEED}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --guidance_scale "${GUIDANCE_SCALE}" \
      --torch_dtype "${TORCH_DTYPE}" \
      --device cuda \
      --attention_backend "${ATTENTION_BACKEND}" \
      --output_dir "${output_dir}" \
      "${EXTRA_ADAPTER_ARGS[@]}" \
      > "${shard_log}" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[adapter] failed ${name}; see ${LOG_DIR}/prompt_adapter_${name}_*.log" >&2
    exit 1
  fi
  echo "[adapter] done ${name}: ${output_dir}"
}

run_one "porn_mlp_depth2_ce" "dataset/baseline_eval/porn_level_4_5.csv" "porn" "${PORN_CKPT}"
run_one "gore_mlp_depth2_ce" "dataset/baseline_eval/gore_level_4_5.csv" "gore" "${GORE_CKPT}"
run_one "ip_doraemon_mlp_depth3_hinge" "dataset/baseline_eval/ip_by_category/ip_doraemon.csv" "ip" "${IP_CKPT}"
run_one "ip_elsa_mlp_depth3_hinge" "dataset/baseline_eval/ip_by_category/ip_elsa.csv" "ip" "${IP_CKPT}"
run_one "ip_minions_mlp_depth3_hinge" "dataset/baseline_eval/ip_by_category/ip_minions.csv" "ip" "${IP_CKPT}"
run_one "ip_snow_white_mlp_depth3_hinge" "dataset/baseline_eval/ip_by_category/ip_snow_white.csv" "ip" "${IP_CKPT}"
run_one "ip_spongebob_squarepants_mlp_depth3_hinge" "dataset/baseline_eval/ip_by_category/ip_spongebob_squarepants.csv" "ip" "${IP_CKPT}"

echo "[adapter] all done"
