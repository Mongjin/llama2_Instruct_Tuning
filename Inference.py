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

args.output_dir = "llama-7-int4-dolly"

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

print(sample)

prompt = f"""### Instruction:
너는 dialogue state를 고려하면서 사용자의 추천 요청 질의에 응답하는 챗봇이야. dialogue state는 주어진 대화를 기반으로 알맞게 생성해줘. 내가 전체 대화를 주면 마지막 turn까지 진행된 dialogue state를 key-value 형태로 [Output State] 뒤에 생성해줘, 그리고 더 적합한 추천을 하기 위해서 필요한 정보를 추가적으로 채워야하는 (Required Additional State) dialogue state도 key-value 형식으로 [Output State] 뒤에 생성해줘. 마지막으로 [Output Response] 뒤에 마지막 user 발화에 알맞은 답변도 생성해줘.

### Input:
[State] None [Dialogue] user: 이따 저녁에 뭐 먹을지 고민이네

### Response: [Output State]
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)

# print(f"Prompt:\n{sample['response']}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
# print(f"Ground truth:\n{sample['instruction']}")
