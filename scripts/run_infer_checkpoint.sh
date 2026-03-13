#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_dir> [text] [speaker_name] [output_wav] [language]"
  exit 1
fi

CHECKPOINT_DIR="$1"
TEXT="${2:-Salam, bu səs modelinin test cümləsidir.}"
SPEAKER_NAME="${3:-azerbaijani_speaker}"
OUTPUT_WAV="${4:-output_test.wav}"
LANGUAGE="${5:-${LANGUAGE:-azerbaijani}}"
DEVICE="${DEVICE:-cpu}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
DO_SAMPLE="${DO_SAMPLE:-true}"
SUBTALKER_DOSAMPLE="${SUBTALKER_DOSAMPLE:-${DO_SAMPLE}}"
TEMPERATURE="${TEMPERATURE:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"

python3 - <<PY
import torch
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel

checkpoint_dir = "${CHECKPOINT_DIR}"
text = """${TEXT}"""
speaker = "${SPEAKER_NAME}"
output_wav = "${OUTPUT_WAV}"
language = "${LANGUAGE}"
device = "${DEVICE}"
max_new_tokens = int("${MAX_NEW_TOKENS}")
do_sample = "${DO_SAMPLE}".strip().lower() == "true"
subtalker_dosample = "${SUBTALKER_DOSAMPLE}".strip().lower() == "true"
temperature = float("${TEMPERATURE}")
repetition_penalty = float("${REPETITION_PENALTY}")

use_cuda = device.startswith("cuda")
use_mps = device == "mps"
attn_impl = "flash_attention_2" if use_cuda else "eager"
model_dtype = torch.bfloat16 if use_cuda else torch.float32

print(f"Loading checkpoint: {checkpoint_dir}")
tts = Qwen3TTSModel.from_pretrained(
    checkpoint_dir,
    device_map=device,
    dtype=model_dtype,
    attn_implementation=attn_impl,
)

print(f"Generating speech (max_new_tokens={max_new_tokens}, do_sample={do_sample})...")
wavs, sr = tts.generate_custom_voice(
    text=text,
    speaker=speaker,
    language=language,
  max_new_tokens=max_new_tokens,
  do_sample=do_sample,
  subtalker_dosample=subtalker_dosample,
  temperature=temperature,
  repetition_penalty=repetition_penalty,
)

if not wavs:
  raise RuntimeError("Model returned no waveform.")

audio = wavs[0]
if audio is None:
  raise RuntimeError("Model returned None waveform.")

audio = np.asarray(audio)
if audio.size == 0:
  raise RuntimeError("Model returned empty waveform (0 samples).")

if audio.ndim > 1:
  audio = audio.mean(axis=-1)

audio = audio.astype(np.float32)
peak = np.max(np.abs(audio))
if peak > 1.0:
  audio = audio / peak

duration_sec = audio.shape[0] / float(sr)
print(f"Waveform stats: samples={audio.shape[0]}, sr={sr}, duration={duration_sec:.2f}s, peak={float(np.max(np.abs(audio))):.4f}")

sf.write(output_wav, audio, sr, subtype="PCM_16")
print(f"Saved {output_wav}")
PY
