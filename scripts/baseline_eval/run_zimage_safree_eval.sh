#!/usr/bin/env bash
set -euo pipefail

# SAFREE baseline：统一使用 toxic0 target concepts。
# 默认跑 porn、gore 和 5 个 IP；如果只想 smoke test，可以用 NUM_PROMPTS=20 覆盖全量设置。

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CONDA_ENV="${CONDA_ENV:-loraretrieval}"
DEVICE_IDS="${DEVICE_IDS:-0,1,2,3}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-9}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.0}"
SAMPLE_SEED="${SAMPLE_SEED:-20260715}"
GENERATION_SEED="${GENERATION_SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baselines/zimage_safree}"
LOG_DIR="${LOG_DIR:-outputs/logs/baseline_eval}"
SAFREE_PRESET="${SAFREE_PRESET:-toxic0}"
SF_ALPHA="${SF_ALPHA:-0.01}"
SAFREE_SCALE="${SAFREE_SCALE:-1.0}"
RE_ATTN_T="${RE_ATTN_T:--1,100000}"

mkdir -p "${LOG_DIR}"

PROMPT_SELECTION_ARGS=(--use_all_prompts)
if [[ -n "${NUM_PROMPTS:-}" ]]; then
  PROMPT_SELECTION_ARGS=(--num_prompts "${NUM_PROMPTS}")
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

COMPARE_ARGS=()
if [[ "${SAVE_COMPARE:-0}" == "1" ]]; then
  COMPARE_ARGS=(--run_base --save_compare)
fi

run_one() {
  local name="$1"
  local input_csv="$2"
  local output_dir="${OUTPUT_ROOT}/${name}_${SAFREE_PRESET}_seed${GENERATION_SEED}"
  local log_file="${LOG_DIR}/zimage_safree_${name}_${SAFREE_PRESET}.log"

  if [[ -f "${output_dir}/summary.json" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[safree] skip ${name}: ${output_dir}/summary.json already exists; set OVERWRITE=1 to rerun"
    return
  fi

  echo "[safree] start ${name}: ${input_csv}"
  conda run -n "${CONDA_ENV}" python -u baseline/zimage_safree_runner/run_zimage_safree_baseline.py \
    --input_csv "${input_csv}" \
    --output_dir "${output_dir}" \
    --unsafe_preset "${SAFREE_PRESET}" \
    "${PROMPT_SELECTION_ARGS[@]}" \
    --sample_seed "${SAMPLE_SEED}" \
    --generation_seed "${GENERATION_SEED}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --sf_alpha "${SF_ALPHA}" \
    --safree_scale "${SAFREE_SCALE}" \
    --re_attn_t="${RE_ATTN_T}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device_ids "${DEVICE_IDS}" \
    --attention_backend "${ATTENTION_BACKEND}" \
    "${COMPARE_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}" \
    > "${log_file}" 2>&1
  echo "[safree] done ${name}: ${output_dir}"
}

run_one "porn" "dataset/baseline_eval/porn_level_4_5.csv"
run_one "gore" "dataset/baseline_eval/gore_level_4_5.csv"
run_one "ip_doraemon" "dataset/baseline_eval/ip_by_category/ip_doraemon.csv"
run_one "ip_elsa" "dataset/baseline_eval/ip_by_category/ip_elsa.csv"
run_one "ip_minions" "dataset/baseline_eval/ip_by_category/ip_minions.csv"
run_one "ip_snow_white" "dataset/baseline_eval/ip_by_category/ip_snow_white.csv"
run_one "ip_spongebob_squarepants" "dataset/baseline_eval/ip_by_category/ip_spongebob_squarepants.csv"

echo "[safree] all done"
