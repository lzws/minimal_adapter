#!/usr/bin/env bash
set -euo pipefail

# 当前已经生成的 base / SAFREE / STG / adapter 图片的统一定量评测入口。
# 该脚本只调用工作目录内的评测工具，不修改外部 label_image_0629.py。

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONDA_ENV="${CONDA_ENV:-loraretrieval}"
LABEL_SCRIPT="${LABEL_SCRIPT:-/mnt/nas2/zhiwen/SafeGuard/safree_0615/label_image_0629.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/quant_eval}"
MAX_WORKERS="${MAX_WORKERS:-16}"
MODEL="${MODEL:-qwen3.7-plus}"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "[quant-eval] 未设置 DASHSCOPE_API_KEY，无法调用图片审核 API。" >&2
  exit 1
fi

run_one() {
  local name="$1"
  local task="$2"
  local input_dir="$3"
  local metadata_csv="$4"
  local target_ip="${5:-}"

  if [[ ! -d "${input_dir}" ]]; then
    echo "[quant-eval] skip ${name}: 图片目录不存在 ${input_dir}"
    return
  fi

  local cmd=(
    conda run -n "${CONDA_ENV}" python -u
    scripts/quant_eval/label_and_summarize_images.py
    --name "${name}"
    --task "${task}"
    --input_dir "${input_dir}"
    --metadata_csv "${metadata_csv}"
    --label_script "${LABEL_SCRIPT}"
    --output_root "${OUTPUT_ROOT}"
    --model "${MODEL}"
    --max_workers "${MAX_WORKERS}"
  )
  if [[ -n "${target_ip}" ]]; then
    cmd+=(--target_ip "${target_ip}")
  fi

  echo "[quant-eval] start ${name}"
  "${cmd[@]}"
  echo "[quant-eval] done  ${name}"
}

# 可以用 TASKS=porn,gore,ip,benign 选择任务；默认评测当前已有的全部风险实验。
TASKS="${TASKS:-porn,gore,ip}"
IFS=',' read -r -a TASK_LIST <<< "${TASKS}"

for task in "${TASK_LIST[@]}"; do
  case "${task}" in
    porn)
      run_one "base_porn" "porn" \
        "outputs/baselines/zimage_base/porn_seed42/images" \
        "dataset/baseline_eval/porn_level_4_5.csv"
      run_one "safree_porn" "porn" \
        "outputs/baselines/zimage_safree/porn_toxic0_seed42/images" \
        "dataset/baseline_eval/porn_level_4_5.csv"
      run_one "stg_clip_porn" "porn" \
        "outputs/baselines/zimage_stg_clip/porn_toxic0_seed42/images" \
        "dataset/baseline_eval/porn_level_4_5.csv"
      run_one "adapter_porn" "porn" \
        "outputs/baselines/prompt_adapter/porn_mlp_depth2_ce_seed42/adapter" \
        "dataset/baseline_eval/porn_level_4_5.csv"
      ;;
    gore)
      run_one "base_gore" "gore" \
        "outputs/baselines/zimage_base/gore_seed42/images" \
        "dataset/baseline_eval/gore_level_4_5.csv"
      run_one "safree_gore" "gore" \
        "outputs/baselines/zimage_safree/gore_toxic0_seed42/images" \
        "dataset/baseline_eval/gore_level_4_5.csv"
      run_one "stg_clip_gore" "gore" \
        "outputs/baselines/zimage_stg_clip/gore_toxic0_seed42/images" \
        "dataset/baseline_eval/gore_level_4_5.csv"
      run_one "adapter_gore" "gore" \
        "outputs/baselines/prompt_adapter/gore_mlp_depth2_ce_seed42/adapter" \
        "dataset/baseline_eval/gore_level_4_5.csv"
      ;;
    ip)
      for ip_name in doraemon elsa minions snow_white spongebob_squarepants; do
        case "${ip_name}" in
          doraemon) display_name="Doraemon" ;;
          elsa) display_name="Elsa" ;;
          minions) display_name="Minions" ;;
          snow_white) display_name="Snow White" ;;
          spongebob_squarepants) display_name="Spongebob Squarepants" ;;
        esac
        metadata_csv="dataset/baseline_eval/ip_by_category/ip_${ip_name}.csv"
        run_one "base_ip_${ip_name}" "ip" \
          "outputs/baselines/zimage_base/ip_${ip_name}_seed42/images" \
          "${metadata_csv}" "${display_name}"
        run_one "safree_ip_${ip_name}" "ip" \
          "outputs/baselines/zimage_safree/ip_${ip_name}_toxic0_seed42/images" \
          "${metadata_csv}" "${display_name}"
        run_one "stg_clip_ip_${ip_name}" "ip" \
          "outputs/baselines/zimage_stg_clip/ip_${ip_name}_toxic0_seed42/images" \
          "${metadata_csv}" "${display_name}"
        run_one "adapter_ip_${ip_name}" "ip" \
          "outputs/baselines/prompt_adapter/ip_${ip_name}_mlp_depth3_hinge_seed42/adapter" \
          "${metadata_csv}" "${display_name}"
      done
      ;;
    benign)
      echo "[quant-eval] 当前生成目录中没有单独的 benign 结果，需先生成后再运行。"
      ;;
    *)
      echo "[quant-eval] 未知 TASKS 项: ${task}" >&2
      exit 1
      ;;
  esac
done

conda run -n "${CONDA_ENV}" python -u scripts/quant_eval/aggregate_summaries.py \
  --summary_root "${OUTPUT_ROOT}" \
  --output_csv "${OUTPUT_ROOT}/comparison.csv" \
  --output_json "${OUTPUT_ROOT}/comparison.json"

echo "[quant-eval] all done: ${OUTPUT_ROOT}/comparison.csv"
