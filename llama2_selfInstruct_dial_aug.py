use_flash_attention = False
if use_flash_attention:
    # unpatch flash attention
    from utils.llama_patch import unplace_flash_attn_with_attn
    unplace_flash_attn_with_attn()

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments
import json

model_id = "NousResearch/Llama-2-7b-chat-hf" # non gated with RLHF version

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

datas = []
with open('samples_translation.json', 'r', encoding='r') as fr:
    for line in fr.readlines():
        datas.append(json.loads(line))

print(len(datas))
# prompt = f"""### Instruction:
# You are a chatbot that responds to user queries for recommendations while considering the dialogue state. The dialogue state should be appropriately generated based on the given conversation. If I provide the entire conversation, create the dialogue state up to the last turn in a key-value format after [Output State]. Additionally, to make a more suitable recommendation, create the required additional dialogue state in a key-value format after [Output State] as well. Finally, after [Output Response], generate an appropriate response to the last user utterance.
#
# ### Input:
# [State] None [Dialogue] user: I'm thinking about what to eat for dinner later.
#
# ### Response: [Output State]
# """
#
# input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# # with torch.inference_mode():
# outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)
#
# # print(f"Prompt:\n{sample['response']}\n")
# print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
# # print(f"Ground truth:\n{sample['instruction']}")
