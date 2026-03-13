#!/usr/bin/env python3

import argparse
import io
import json
from pathlib import Path
from urllib.request import urlopen

import librosa
import soundfile as sf
from datasets import Audio, load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="ughurabbasov/azerbaijani-tts-dataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="workdir")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--ref_index", type=int, default=0)
    parser.add_argument("--sampling_rate", type=int, default=24000)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--language", type=str, default="azerbaijani")
    return parser.parse_args()


def main():
    args = parse_args()

    out_root = Path(args.output_dir).resolve()
    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl_path = out_root / "train_raw.jsonl"
    ref_audio_path = out_root / "ref.wav"

    dataset = load_dataset(args.dataset_id, split=args.split, token=args.hf_token)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    total = len(dataset)
    limit = args.max_samples if args.max_samples > 0 else total
    limit = min(limit, total)

    if args.ref_index < 0 or args.ref_index >= limit:
        raise ValueError(f"ref_index={args.ref_index} is out of range for {limit} samples")

    def load_audio_dict(audio_obj):
        if isinstance(audio_obj, dict) and "array" in audio_obj and "sampling_rate" in audio_obj:
            waveform = audio_obj["array"]
            sample_rate = int(audio_obj["sampling_rate"])
        elif isinstance(audio_obj, dict):
            audio_path = audio_obj.get("path")
            audio_bytes = audio_obj.get("bytes")

            if audio_bytes is not None:
                waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            elif audio_path:
                if str(audio_path).startswith("http://") or str(audio_path).startswith("https://"):
                    with urlopen(audio_path) as response:
                        payload = response.read()
                    waveform, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
                else:
                    waveform, sample_rate = sf.read(audio_path, dtype="float32")
            else:
                raise ValueError("Unsupported audio object: missing both path and bytes")
        else:
            raise ValueError("Unsupported audio format in dataset")

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        sample_rate = int(sample_rate)
        if sample_rate != args.sampling_rate:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=args.sampling_rate)
            sample_rate = args.sampling_rate

        return waveform, sample_rate

    ref_wave, ref_sr = load_audio_dict(dataset[args.ref_index]["audio"])
    sf.write(ref_audio_path.as_posix(), ref_wave, ref_sr)

    with raw_jsonl_path.open("w", encoding="utf-8") as writer:
        for idx in range(limit):
            row = dataset[idx]
            text = (row.get("text") or "").strip()
            if not text:
                continue

            audio = row["audio"]
            wave, sr = load_audio_dict(audio)
            wav_path = audio_dir / f"utt{idx:06d}.wav"
            sf.write(wav_path.as_posix(), wave, sr)

            item = {
                "audio": wav_path.as_posix(),
                "text": text,
                "ref_audio": ref_audio_path.as_posix(),
                "language": args.language.lower(),
            }
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved raw jsonl: {raw_jsonl_path}")
    print(f"Saved reference audio: {ref_audio_path}")
    print(f"Saved utterances under: {audio_dir}")


if __name__ == "__main__":
    main()
