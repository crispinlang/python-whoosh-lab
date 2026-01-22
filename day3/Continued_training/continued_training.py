import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

max_seq_length = 1024
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    bias="none",
    lora_alpha = 16,
    lora_dropout = 0,
    random_state=1234,
    use_rslora=False,
    loftq_config=None,
    use_gradient_checkpointing = "unsloth",
)

lm_prompt = """{}"""
EOS_TOKEN = tokenizer.eos_token

import re

def format_dataset_lm():
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text_data = f.read()

    raw_chunks = re.split(r'\n\s*\n', text_data)

    formatted_texts = []
    for chunk in raw_chunks:
        clean_chunk = chunk.strip()
        
        if len(clean_chunk) > 5:
            training_example = lm_prompt.format(clean_chunk) + EOS_TOKEN
            
            formatted_texts.append(training_example)

    dataset = Dataset.from_dict({"text": formatted_texts})
    
    print(f"Found {len(formatted_texts)} examples.")
    print("--- Example 1 Preview ---")
    print(dataset[50:55]["text"])
    print("-------------------------")
    
    return dataset

dataset = format_dataset_lm()

# Config the trainer
sftConfig = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps = 100,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=1234,
    output_dir="outputs",
)


# Trainer object
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = sftConfig
)

t_obj = trainer.train()

# Switch to inference mode
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    lm_prompt.format("albert: ")
], return_tensors = "pt").to("cuda")


text_streamer = TextStreamer(tokenizer)

_ = model.generate(**inputs, 
                   streamer = text_streamer, 
                   min_new_tokens = 50,
                   max_new_tokens = 100,
                   repetition_penalty = 1.1,
                   #temperature = 0.7
                   )