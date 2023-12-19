from datasets import load_dataset
from random import randrange
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments
import json

use_flash_attention = False
if use_flash_attention:
    # unpatch flash attention
    from utils.llama_patch import unplace_flash_attn_with_attn
    unplace_flash_attn_with_attn()

# args = TrainingArguments(
#     output_dir="Llama-2-13b-DST-seed-only",
#     num_train_epochs=3,
#     per_device_train_batch_size=6 if use_flash_attention else 4,
#     gradient_accumulation_steps=2,
#     gradient_checkpointing=True,
#     optim="paged_adamw_32bit",
#     logging_steps=10,
#     save_strategy="epoch",
#     learning_rate=2e-4,
#     bf16=True,
#     tf32=True,
#     max_grad_norm=0.3,
#     warmup_ratio=0.03,
#     lr_scheduler_type="constant",
#     disable_tqdm=True # disable tqdm since with packing values are in correct
# )

# args.output_dir = "NousResearch/Llama-2-7b-chat-hf"
#
# # load base LLM model and tokenizer
# model = AutoPeftModelForCausalLM.from_pretrained(
#     args.output_dir,
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.float16,
#     load_in_4bit=True,
# )

def get_dst_instruction_data(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            datas.append(json.loads(line))
    return datas


def format_instruction(sample):
    dialogue = sample['dialogue']
    sample['prev_state'] = "None"
    if "bot: " in dialogue:
        bot_index = dialogue.index("bot:")
        # len("abot: ") = 6, "abot: ".rindex("bot: ") = 1
        dialogue = dialogue[:bot_index]
    return f"""### Instruction: Update 'cur_state' (i.e., current state) based on last user's utterance of [Dialogue]. Follow tese rules: First, if there are no additional information to update 'cur_state', you can just output same content as 'prev_state'. Second, update dialogue states of given dialogue. Third, do not generate additional utterances or explain. Please update 'cur_state' while considering these factors. \n ### Input: [Previous state] 'prev_state': {sample['prev_state']} [Dialogue] {dialogue} \n ### Output: [Current state] """

dataset = get_dst_instruction_data('./data/samples_translation.json')
model_id = "Llama-2-13b-DST-v2" # non gated with RLHF version

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


# Load dataset from the hub and get a sample
# dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
# sample = dataset[randrange(len(dataset))]

# print(sample)

results = []

for i, data in enumerate(dataset):
    prompt = format_instruction(data)
    print(f"""prompt: \n {prompt}""")
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.1)

    # print(f"Prompt:\n{sample['response']}\n")
    result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    print(
        f"Generated instruction:\n{result}")
    # print(f"Ground truth:\n{sample['instruction']}")
    data['generated_state'] = result
    results.append(data)

with open('./results/close_domain_results.json', 'w', encoding='utf-8') as fw:
    for result in results:
        fw.write(json.dumps(result, ensure_ascii=False) + "\n")
