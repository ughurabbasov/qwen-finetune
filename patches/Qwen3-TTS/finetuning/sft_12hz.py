import argparse
import json
import math
import os
import shutil
import time

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def _configure_cpu_thread_limit(max_utilization=0.5):
    total_cpus = os.cpu_count() or 1
    limited_threads = max(1, int(total_cpus * max_utilization))

    os.environ["OMP_NUM_THREADS"] = str(limited_threads)
    os.environ["MKL_NUM_THREADS"] = str(limited_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(limited_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(limited_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(limited_threads)

    torch.set_num_threads(limited_threads)
    torch.set_num_interop_threads(max(1, limited_threads // 2))

    return limited_threads, total_cpus


def _resolve_local_model_path(path_value, project_root):
    candidates = []

    if os.path.isabs(path_value):
        candidates.append(os.path.abspath(path_value))
    else:
        candidates.append(os.path.abspath(path_value))
        candidates.append(os.path.abspath(os.path.join(project_root, path_value)))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    checked = "\n  - ".join(candidates)
    raise FileNotFoundError(
        f"init_model_path must be a local folder. Checked:\n  - {checked}"
    )


def train():
    global target_speaker_embedding

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    default_init_model_path = os.path.join(project_root, "Qwen3-TTS-12Hz-0.6B-Base")

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default=default_init_model_path)
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--log_every_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--new_language_name", type=str, default="")
    parser.add_argument("--new_language_id", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logging_dir", type=str, default=None)
    args = parser.parse_args()

    output_model_path = args.output_model_path
    if not os.path.isabs(output_model_path):
        output_model_path = os.path.abspath(os.path.join(project_root, output_model_path))
    os.makedirs(output_model_path, exist_ok=True)

    MODEL_PATH = _resolve_local_model_path(args.init_model_path, project_root)

    input_config_file = os.path.join(MODEL_PATH, "config.json")
    if not os.path.isfile(input_config_file):
        raise FileNotFoundError(f"Missing config.json in init model folder: {MODEL_PATH}")

    use_cpu = args.device.strip().lower() == "cpu"
    use_gpu = args.device.startswith("cuda")
    mixed_precision = "bf16" if use_gpu else "no"
    attn_impl = "sdpa" if use_gpu else "eager"
    model_dtype = torch.bfloat16 if use_gpu else torch.float32

    if use_cpu:
        limited_threads, total_cpus = _configure_cpu_thread_limit(max_utilization=0.5)
        print(f"CPU mode: limiting threads to {limited_threads}/{total_cpus} (~50%).")

    logging_dir = args.logging_dir or os.path.join(output_model_path, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
        cpu=use_cpu,
    )

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cpu" if use_cpu else args.device,
        dtype=model_dtype,
        attn_implementation=attn_impl,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    new_language_name = args.new_language_name.strip().lower()
    if new_language_name:
        codec_language_id = dict(config.talker_config.codec_language_id)
        if args.new_language_id >= 0:
            language_id = int(args.new_language_id)
        elif new_language_name in codec_language_id:
            language_id = int(codec_language_id[new_language_name])
        else:
            language_id = max(int(v) for v in codec_language_id.values()) + 1

        if language_id >= int(config.talker_config.vocab_size):
            raise ValueError(
                f"new_language_id={language_id} is out of talker vocab range "
                f"(vocab_size={config.talker_config.vocab_size})."
            )

        codec_language_id[new_language_name] = language_id
        config.talker_config.codec_language_id = codec_language_id

        model_codec_language_id = dict(qwen3tts.model.config.talker_config.codec_language_id)
        model_codec_language_id[new_language_name] = language_id
        qwen3tts.model.config.talker_config.codec_language_id = model_codec_language_id
        print(f"Registered language '{new_language_name}' with codec id {language_id}.")

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    print(
        f"Loaded dataset: {len(dataset)} samples | batch_size={args.batch_size} | "
        f"grad_accum={args.gradient_accumulation_steps} | device={args.device}",
        flush=True,
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    num_epochs = args.num_epochs
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_training_steps = steps_per_epoch * num_epochs
    if args.max_train_steps > 0:
        total_training_steps = min(total_training_steps, args.max_train_steps)
    warmup_steps = args.warmup_steps

    scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=1e-6)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, scheduler
    )

    if accelerator.is_main_process:
        print(
            f"Training starts: epochs={num_epochs}, steps/epoch={steps_per_epoch}, "
            f"total_steps={total_training_steps}, warmup_steps={warmup_steps}",
            flush=True,
        )

    global_step = 0
    model.train()

    # Rolling step checkpoint dir — overwritten each time to save storage
    step_ckpt_dir = os.path.join(output_model_path, "checkpoint-latest")

    def _save_checkpoint(output_dir, copy_base_files):
        os.makedirs(output_dir, exist_ok=True)

        if copy_base_files:
            try:
                shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)
            except Exception as exc:
                accelerator.print(f"Warning: failed to copy base model files to {output_dir}: {exc}")

        with open(input_config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {
            args.speaker_name: 3000
        }
        talker_config["spk_is_dialect"] = {
            args.speaker_name: False
        }
        talker_config["codec_language_id"] = dict(config.talker_config.codec_language_id)
        config_dict["talker_config"] = talker_config

        output_config_file = os.path.join(output_dir, "config.json")
        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

        drop_prefix = "speaker_encoder"
        keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
        for k in keys_to_drop:
            del state_dict[k]

        if target_speaker_embedding is not None:
            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        else:
            accelerator.print("Warning: target speaker embedding is not initialized; keeping default speaker row 3000.")

        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        accelerator.print(f"Saved checkpoint to {output_dir}")

    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch} started.", flush=True)
        for step, batch in enumerate(train_dataloader):
            micro_step_start = time.perf_counter()
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']
                speaker_positions = batch["speaker_positions"]

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.text_projection(
                    model.talker.get_text_embeddings()(input_text_ids)
                ) * text_embedding_mask
                input_codec_embedding = model.talker.get_input_embeddings()(input_codec_ids) * codec_embedding_mask
                speaker_positions = speaker_positions.to(input_codec_embedding.device)
                batch_indices = torch.arange(input_codec_embedding.shape[0], device=input_codec_embedding.device)
                input_codec_embedding[batch_indices, speaker_positions, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    shift_labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                sub_mask = codec_mask[:, 1:]
                talker_hidden_states = hidden_states[sub_mask]
                talker_codec_ids = codec_ids[:, 1:, :][sub_mask]

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # Linear warmup then cosine decay
                if global_step < warmup_steps:
                    warmup_factor = (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg['lr'] = args.lr * warmup_factor
                else:
                    scheduler.step()

                optimizer.zero_grad()

            current_lr = optimizer.param_groups[0]['lr']
            step_time = time.perf_counter() - micro_step_start
            if accelerator.is_main_process and step % max(1, args.log_every_steps) == 0:
                print(
                    f"Epoch {epoch} | Step {step} | GlobalStep {global_step} | "
                    f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | "
                    f"StepTime: {step_time:.2f}s",
                    flush=True,
                )

            global_step += 1
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    print(
                        f"Reached max_train_steps={args.max_train_steps}. Stopping early.",
                        flush=True,
                    )
                break

            if accelerator.is_main_process and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                _save_checkpoint(step_ckpt_dir, copy_base_files=False)

        if accelerator.is_main_process:
            epoch_dir = os.path.join(output_model_path, f"checkpoint-epoch-{epoch}")
            _save_checkpoint(epoch_dir, copy_base_files=True)

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

if __name__ == "__main__":
    train()
