#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${ROOT_DIR}/training.log"
LANGUAGE_NAME="${LANGUAGE_NAME:-azerbaijani}"
LANGUAGE_ID="${LANGUAGE_ID:-2075}"

cd "${ROOT_DIR}"
source .venv/bin/activate

echo "Starting training in background. Logs: ${LOG_FILE}"
echo "Monitor with: tail -f ${LOG_FILE}"
echo "Check process: ps aux | grep sft_12hz.py"

nohup caffeinate -i python3 Qwen3-TTS/finetuning/sft_12hz.py \
  --device cpu \
  --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path output_0_6b \
  --train_jsonl workdir/train_with_codes.jsonl \
  --batch_size 1 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name azerbaijani_speaker \
  --new_language_name "${LANGUAGE_NAME}" \
  --new_language_id "${LANGUAGE_ID}" \
  > "${LOG_FILE}" 2>&1 &

echo "Training PID: $!"
echo "Started successfully. Monitor with: tail -f ${LOG_FILE}"
