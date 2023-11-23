from datasets import load_dataset
from random import randrange
import json

# Load dataset from the hub
# dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
#
# print(f"dataset size: {len(dataset)}")
# print(dataset[randrange(len(dataset))])
# dataset size: 15011


def get_dst_instruction_data(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            datas.append(json.loads(line))
    return datas


def format_instruction(sample):
    dialogue = sample['dialogue']
    if "bot: " in dialogue:
        bot_index = dialogue.rindex("bot: ")
        # len("abot: ") = 6, "abot: ".rindex("bot: ") = 1
        if bot_index == len(dialogue) - 5:
            dialogue = dialogue[:bot_index]
    return f"""### Instruction: Update 'cur_state' (i.e., current state) based on last user's utterance of [Dialogue]. Follow tese rules: First, if there are no additional information to update 'cur_state', you can just output same content as 'prev_state'. Second, update dialogue states of given dialogue. Third, do not generate additional utterances or explain. Please update 'cur_state' while considering these factors. \n ### Input: [Previous state] 'prev_state': {sample['prev_state']} [Dialogue] {dialogue} \n ### Output: [Current state] 'current_state': {sample['cur_state']} """
    # return f"""### Instruction:
    # Use the Input below to create an instruction, which could have been used to generate the input using an LLM.
    #
    # ### Input:
    # {sample['response']}
    #
    # ### Response:
    # {sample['instruction']}
    # """

from random import randrange

dataset = get_dst_instruction_data('./samples_translation.json')
# dataset = []
# for i in range(len(datas)):
#     dataset.append(format_instruction(datas[i]))
#
# print(dataset[0])

print(format_instruction(dataset[randrange(len(dataset))]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

use_flash_attention = False

# Hugging Face model id
# model_id = "NousResearch/Llama-2-7b-hf"  # non-gated
# model_id = "meta-llama/Llama-2-7b-hf" # gated
# model_id = "meta-llama/Llama-2-7b-chat-hf" # gated
# model_id = "NousResearch/Llama-2-7b-chat-hf" # non gated with RLHF version
# model_id = "/home/konkuk/Llama/Llama-2-13b-chat-hf" # non gated with RLHF version
model_id = "/workspace/Llama-2-13b-chat-hf"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=use_flash_attention,
    device_map="auto",
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
)


# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="Llama-2-13b-DST-seed-only",
    num_train_epochs=10,
    per_device_train_batch_size=12 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True # disable tqdm since with packing values are in correct
)

from trl import SFTTrainer

max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)

# train
trainer.train() # there will not be a progress bar since tqdm is disabled

# save model
trainer.save_model()

