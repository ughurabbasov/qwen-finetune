# Qwen3-TTS 0.6B Finetuning (Azerbaijani)

This repo provides a streamlined, script-driven workflow to finetune
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` using the upstream Qwen3-TTS finetuning code and
the Hugging Face dataset `ughurabbasov/azerbaijani-tts-dataset`.

The main entry point is `scripts/run_finetune_0_6b.sh`, which runs:

1) dataset -> JSONL preparation
2) audio code extraction
3) SFT finetuning

## Prerequisites

- Python 3.9+ (3.10 recommended)
- Git
- Linux + NVIDIA GPU recommended for training (CPU works but is slow)
- macOS users can try `DEVICE=mps`
- If `soundfile` fails, install `libsndfile` (macOS: `brew install libsndfile`)

## Quickstart

### 1) Clone upstream finetuning code

The scripts expect the official Qwen3-TTS repo at `Qwen3-TTS/`:

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git Qwen3-TTS
```

### 2) Apply local fixes

This repo includes patched files under `patches/Qwen3-TTS/`. Sync them into the
cloned repo:

```bash
rsync -a patches/Qwen3-TTS/ Qwen3-TTS/
```

### 3) Create and activate the venv

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

### 4) Run full finetuning

```bash
DEVICE=cuda:0 bash scripts/run_finetune_0_6b.sh
```

For macOS:

```bash
DEVICE=mps bash scripts/run_finetune_0_6b.sh
```

## What the script does

`scripts/run_finetune_0_6b.sh` executes three stages:

1) **Prepare JSONL** using `scripts/prepare_az_tts_data.py`:
   - Downloads the HF dataset
   - Saves `workdir/train_raw.jsonl`
   - Writes per-utterance WAVs to `workdir/audio/`
   - Creates a single `workdir/ref.wav` used as `ref_audio` for all samples
2) **Extract audio codes** with `Qwen3-TTS/finetuning/prepare_data.py`:
   - Produces `workdir/train_with_codes.jsonl`
3) **SFT finetune** with `Qwen3-TTS/finetuning/sft_12hz.py`:
   - Writes checkpoints to `output_0_6b/checkpoint-epoch-*`

## Configuration

All knobs are passed via env vars to `scripts/run_finetune_0_6b.sh`:

```bash
DEVICE=cuda:0 \
TOKENIZER_MODEL_PATH=Qwen/Qwen3-TTS-Tokenizer-12Hz \
INIT_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
HF_DATASET_ID=ughurabbasov/azerbaijani-tts-dataset \
HF_SPLIT=train \
HF_TOKEN=hf_xxx \
MAX_SAMPLES=0 \
REF_INDEX=0 \
BATCH_SIZE=2 \
LR=2e-5 \
EPOCHS=3 \
SPEAKER_NAME=azerbaijani_speaker \
LANGUAGE_NAME=azerbaijani \
LANGUAGE_ID=2075 \
bash scripts/run_finetune_0_6b.sh
```

Notes:
- `MAX_SAMPLES=0` means "use all samples"
- `REF_INDEX` selects a single reference WAV from the dataset
- `HF_TOKEN` is required only if the dataset or model is gated

## Inference from a checkpoint

```bash
bash scripts/run_infer_checkpoint.sh \
  output_0_6b/checkpoint-epoch-2 \
  "Salam, men Azerbaycan dilinde danisiram." \
  azerbaijani_speaker \
  output_az.wav
```

Optional env overrides:

```bash
DEVICE=cuda:0 \
MAX_NEW_TOKENS=2048 \
TEMPERATURE=0.9 \
bash scripts/run_infer_checkpoint.sh output_0_6b/checkpoint-epoch-2
```

## Dataset expectations

The dataset must expose `audio` and `text` fields. The script accepts:
- `audio` as HF `Audio` dicts (`path`/`bytes`) or decoded arrays
- `text` as a non-empty string

If your dataset uses different fields, update `scripts/prepare_az_tts_data.py`.

## Troubleshooting

- **Missing Qwen3-TTS**: run the clone step above.
- **HF auth errors**: export `HF_TOKEN=hf_xxx` and retry.
- **`soundfile` import error**: install `libsndfile`.
- **OOM / slow training**: lower `BATCH_SIZE`, `MAX_SAMPLES`, or use a GPU.

## Repo hygiene (for GitHub)

Large artifacts are intentionally ignored. See `.gitignore`. If you want to
version checkpoints or WAVs, use Git LFS.
