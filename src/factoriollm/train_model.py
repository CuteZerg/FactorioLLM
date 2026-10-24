"""
LoRA fine-tuning (Unsloth, 4-bit) on the v4 dataset.

    # smoke test: 20 steps on a small subsample
    uv run python src/factoriollm/train_model.py --run-name smoke --subsample per-blueprint --max-steps 20
    # pilot: one EN + one RU prompt per blueprint, 1 epoch
    uv run python src/factoriollm/train_model.py --run-name pilot-qwen7b --subsample per-blueprint --epochs 1

Outputs: checkpoints and training.log in outputs/<run-name>/, the adapter in adapters/<run-name>/
(the legacy factorio_lora_model folder is never overwritten).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

# Chat template + the markers that delimit user / assistant turns (for loss masking).
TEMPLATES = {
    "qwen-2.5": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen3-instruct": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
}


class TeeLogger:
    """Mirrors stdout/stderr into a log file (installed in main(), never at import time)."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log = log_file

    def write(self, message):
        self.stream.write(message)
        self.stream.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


def detect_template(model_name: str) -> str:
    name = model_name.lower()
    if "qwen3" in name:
        return "qwen3-instruct"
    if "qwen" in name:
        return "qwen-2.5"
    if "llama-3" in name or "llama3" in name:
        return "llama-3"
    raise ValueError(f"Cannot infer chat template for '{model_name}', pass --chat-template")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def subsample_per_blueprint(records: list) -> list:
    """Deterministically keeps one English and one Russian prompt per canonical blueprint."""
    chosen = {}
    for r in sorted(records, key=lambda r: _hash(r["messages"][1]["content"])):
        meta = r.get("meta") or {}
        key = (meta.get("canonical_id"), meta.get("lang"))
        if key[0] is None:
            raise ValueError("Dataset has no 'meta' field: re-export it with export_dataset.py")
        chosen.setdefault(key, r)
    return list(chosen.values())


def load_records(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="FactorioLLM LoRA fine-tuning")
    p.add_argument("--run-name", required=True)
    p.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
    p.add_argument("--chat-template", default=None, choices=sorted(TEMPLATES), help="Default: inferred from --base-model")
    p.add_argument("--dataset", default="data/dataset_v4.jsonl")
    p.add_argument("--eval-dataset", default="data/dataset_v4_eval.jsonl")
    p.add_argument("--subsample", choices=["none", "per-blueprint"], default="none")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (smoke tests)")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--packing", action="store_true", help="Pack short examples (disables assistant-only loss)")
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--eval-max", type=int, default=200, help="Eval examples used for eval loss")
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def train(args) -> None:
    if getattr(torch.utils, "_pytree", None) and not hasattr(torch.utils._pytree, "register_constant"):
        torch.utils._pytree.register_constant = lambda x: None
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    template = args.chat_template or detect_template(args.base_model)
    output_dir = Path("outputs") / args.run_name
    adapter_dir = Path("adapters") / args.run_name

    print(f"[*] Loading {args.base_model} (4-bit, max_seq_length={args.max_seq_length}, template={template})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=template)

    train_records = load_records(args.dataset)
    if args.subsample == "per-blueprint":
        train_records = subsample_per_blueprint(train_records)
    eval_records = sorted(load_records(args.eval_dataset), key=lambda r: _hash(r["messages"][1]["content"]))[: args.eval_max]
    print(f"[*] Train examples: {len(train_records)} ({args.subsample}), eval examples: {len(eval_records)}")

    def to_dataset(records):
        texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False) for r in records]
        return Dataset.from_dict({"text": texts})

    train_ds, eval_ds = to_dataset(train_records), to_dataset(eval_records)

    config = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=args.packing,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=0.03,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=eval_ds, args=config)

    if not args.packing:
        instruction_part, response_part = TEMPLATES[template]
        trainer = train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)
        check_loss_mask(trainer, tokenizer)

    print("[*] Training...")
    stats = trainer.train()
    print(f"[*] Finished in {stats.metrics['train_runtime']:.0f}s; final train loss {stats.metrics.get('train_loss')}")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (adapter_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "chat_template": template, "train_examples": len(train_records), "metrics": stats.metrics}, indent=1),
        encoding="utf-8",
    )
    print(f"[*] Adapter saved to {adapter_dir}")


def check_loss_mask(trainer, tokenizer) -> None:
    """Fails fast if anything other than the assistant's code contributes to the loss."""
    sample = trainer.train_dataset[0]
    trained = [t for t, label in zip(sample["input_ids"], sample["labels"]) if label != -100]
    text = tokenizer.decode(trained, skip_special_tokens=True).strip()
    print(f"[*] Loss-mask check, trained tokens start with: {text[:80]!r}")
    if not text.startswith("from draftsman") or "You are an expert" in text:
        raise RuntimeError("Loss mask is wrong: trained tokens must be the assistant code only")


def main():
    args = parse_args()
    log_dir = Path("outputs") / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "training.log", "w", encoding="utf-8", buffering=1)
    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)
    try:
        train(args)
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
