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
import random

model_id = "NousResearch/Llama-2-13b-chat-hf" # non gated with RLHF version

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
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

datas = []
with open('samples_translation.json', 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        datas.append(json.loads(line))

augmented_dials = []
with open('dialogue_augment.txt', 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        augmented_dials.append(line.strip())

augmented_states = {}
for i in range(0, 30):
    random_indies = []
    while len(random_indies) < 3:
        index = random.randint(0, 59)
        if index not in random_indies:
            random_indies.append(index)

    prompt = f"""### Instruction: Generate dialogue state given dialogues that 'user' is asking 'bot' for recommendation food or travel. I will give you some samples. The 'prev_state' (i.e., previous state) is the dialogue state that determined before the user's last utterance. The 'cur_state' (i.e., current state) is the dialogue state that determined after the user's last utterance. You should generate both 'prev_state' and 'cur_state', following the structure of given samples. \n ### Input: [Dialogue 1] {datas[random_indies[0]]['dialogue']} 'prev_state': {datas[random_indies[0]]['prev_state']} 'cur_state': {datas[random_indies[0]]['cur_state']} \n [Dialogue 2] {datas[random_indies[1]]['dialogue']} 'prev_state': {datas[random_indies[1]]['prev_state']} 'cur_state': {datas[random_indies[1]]['cur_state']} \n [Dialogue 3] {datas[random_indies[2]]['dialogue']} 'prev_state': {datas[random_indies[2]]['prev_state']} 'cur_state': {datas[random_indies[2]]['cur_state']} \n ### Output: [Dialogue 4] {augmented_dials[i]} """

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.9)
    # print(f"Prompt:\n{sample['response']}\n")
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    augmented_states[i] = output
    print(
        f"Generated {(i+1)}-th output:\n{output}")
    # print(f"Ground truth:\n{sample['instruction']}")

with open('./states_augment.json', 'w', encoding='utf-8') as fw:
    json.dump(augmented_states, fw, indent="\t", ensure_ascii=False)
