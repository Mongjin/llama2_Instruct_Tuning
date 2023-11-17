use_flash_attention = False
if use_flash_attention:
    # unpatch flash attention
    from utils.llama_patch import unplace_flash_attn_with_attn
    unplace_flash_attn_with_attn()

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="llama-7-int4-dolly",
    num_train_epochs=3,
    per_device_train_batch_size=6 if use_flash_attention else 4,
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

args.output_dir = "NousResearch/Llama-2-7b-chat-hf"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

from datasets import load_dataset
from random import randrange


# Load dataset from the hub and get a sample
# dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
# sample = dataset[randrange(len(dataset))]

# print(sample)

prompt = f"""### Instruction:
You are a chatbot that responds to user queries for recommendations while considering the dialogue state. The dialogue state should be appropriately generated based on the given conversation. If I provide the entire conversation, create the dialogue state up to the last turn in a key-value format after [Output State]. Additionally, to make a more suitable recommendation, create the required additional dialogue state in a key-value format after [Output State] as well. Finally, after [Output Response], generate an appropriate response to the last user utterance.

### Input:
[State] None [Dialogue] user: I'm thinking about what to eat for dinner later.

### Response: [Output State]
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)

# print(f"Prompt:\n{sample['response']}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
# print(f"Ground truth:\n{sample['instruction']}")
