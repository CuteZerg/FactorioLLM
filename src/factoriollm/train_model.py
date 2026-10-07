import os
import sys
import torch

# Live TeeLogger for real-time unbuffered logging to file and console
class TeeLogger:
    def __init__(self, filename="training.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger("training.log")
sys.stderr = sys.stdout

if getattr(torch.utils, "_pytree", None) and not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda x: None
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

def train():
    print("[*] Loading Model (Llama-3-8B-Instruct 4-bit)...")
    max_seq_length = 2048 # Adjust as needed (Factorio blueprints can be long)
    dtype = None # Auto detection
    load_in_4bit = True # 4bit quantization to fit in 12GB VRAM

    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # = 0 is optimized
        bias = "none",    # = "none" is optimized
        use_gradient_checkpointing = "unsloth", # "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. Format Dataset
    print("[*] Formatting Dataset...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3", # Match our base model
        mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    if os.path.exists("data/dataset_v3.jsonl"):
        dataset_path = "data/dataset_v3.jsonl"
    elif os.path.exists("data/dataset_v2.jsonl"):
        dataset_path = "data/dataset_v2.jsonl"
    else:
        dataset_path = "data/dataset.jsonl"
    print(f"[*] Loading training dataset from '{dataset_path}'...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 4. Train
    print("[*] Initializing Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 1,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2, # Small batch size for 12GB VRAM
            gradient_accumulation_steps = 4, # Simulate larger batch size
            warmup_steps = 5,
            num_train_epochs = 1, # Run for 1 full epoch over the dataset
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    print("[*] Starting Training...")
    trainer_stats = trainer.train()

    # 5. Save the model
    print(f"[*] Training finished! Time taken: {trainer_stats.metrics['train_runtime']} seconds")
    model.save_pretrained("factorio_lora_model")
    tokenizer.save_pretrained("factorio_lora_model")
    print("[*] LoRA adapters saved to 'factorio_lora_model' folder!")

if __name__ == "__main__":
    train()
