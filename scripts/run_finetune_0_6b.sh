#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINETUNE_DIR="${ROOT_DIR}/Qwen3-TTS/finetuning"
WORK_DIR="${ROOT_DIR}/workdir"
OUTPUT_DIR="${ROOT_DIR}/output_0_6b"

DEVICE="${DEVICE:-cpu}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-Qwen/Qwen3-TTS-Tokenizer-12Hz}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
HF_DATASET_ID="${HF_DATASET_ID:-ughurabbasov/azerbaijani-tts-dataset}"
HF_SPLIT="${HF_SPLIT:-train}"
HF_TOKEN="${HF_TOKEN:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
REF_INDEX="${REF_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3}"
SPEAKER_NAME="${SPEAKER_NAME:-azerbaijani_speaker}"
LANGUAGE_NAME="${LANGUAGE_NAME:-azerbaijani}"
LANGUAGE_ID="${LANGUAGE_ID:-2075}"

mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"

if [[ ! -d "${FINETUNE_DIR}" ]]; then
  echo "Missing ${FINETUNE_DIR}. Clone Qwen3-TTS into ${ROOT_DIR} first."
  exit 1
fi

echo "[1/3] Preparing raw JSONL from Hugging Face dataset..."
if [[ -n "${HF_TOKEN}" ]]; then
  python3 "${ROOT_DIR}/scripts/prepare_az_tts_data.py" \
    --dataset_id "${HF_DATASET_ID}" \
    --split "${HF_SPLIT}" \
    --output_dir "${WORK_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --ref_index "${REF_INDEX}" \
    --language "${LANGUAGE_NAME}" \
    --hf_token "${HF_TOKEN}"
else
  python3 "${ROOT_DIR}/scripts/prepare_az_tts_data.py" \
    --dataset_id "${HF_DATASET_ID}" \
    --split "${HF_SPLIT}" \
    --output_dir "${WORK_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --ref_index "${REF_INDEX}" \
    --language "${LANGUAGE_NAME}"
fi

echo "[2/3] Extracting audio codes..."
python3 "${FINETUNE_DIR}/prepare_data.py" \
  --device "${DEVICE}" \
  --tokenizer_model_path "${TOKENIZER_MODEL_PATH}" \
  --input_jsonl "${WORK_DIR}/train_raw.jsonl" \
  --output_jsonl "${WORK_DIR}/train_with_codes.jsonl"

echo "[3/3] Running SFT finetuning on 0.6B base..."
python3 "${FINETUNE_DIR}/sft_12hz.py" \
  --device "${DEVICE}" \
  --init_model_path "${INIT_MODEL_PATH}" \
  --output_model_path "${OUTPUT_DIR}" \
  --train_jsonl "${WORK_DIR}/train_with_codes.jsonl" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_epochs "${EPOCHS}" \
  --speaker_name "${SPEAKER_NAME}" \
  --new_language_name "${LANGUAGE_NAME}" \
  --new_language_id "${LANGUAGE_ID}"

echo "Finetuning complete. Checkpoints are in ${OUTPUT_DIR}/checkpoint-epoch-*"
