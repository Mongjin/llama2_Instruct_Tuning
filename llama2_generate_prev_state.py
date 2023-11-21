use_flash_attention = False
if use_flash_attention:
    # unpatch flash attention
    from utils.llama_patch import unplace_flash_attn_with_attn
    unplace_flash_attn_with_attn()

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import TrainingArguments
import json
import random

model_id = "NousResearch/Llama-2-13b-chat-hf" # non gated with RLHF version

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = LlamaForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=use_flash_attention,
    device_map="auto"
)
model.config.pretraining_tp = 1
model.eval()

tokenizer = LlamaTokenizer.from_pretrained(model_id)

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
    while len(random_indies) < 4:
        index = random.randint(0, 59)
        if index not in random_indies:
            random_indies.append(index)
    target_dialogue = augmented_dials[i]
    user_utt_index = target_dialogue.rindex('user:')
    if user_utt_index != 0:
        target_dialogue = target_dialogue[:user_utt_index]

    print(target_dialogue)
    prompt = f"""### Instruction: Create dialogue states based on [Dialogue 5]. [Dialogue 1] ~ [Dialogue 4] and corresponding dialogue states will be given as examples and when given [Dialogue 5], create dialogue states. Following these rules: First, as shown in the examples, create dialogue states for better recommendation. Second, do not generate additional utterances. Please create 'prev_state' of [Dialogue 5] while considering these factors. \n ### Input: [Dialogue 1] {datas[random_indies[0]]['dialogue']}[Previous state 1] 'prev_state': {datas[random_indies[0]]['prev_state']} \n [Dialogue 2] {datas[random_indies[1]]['dialogue']} [Previous state 2] 'prev_state': {datas[random_indies[1]]['prev_state']} \n [Dialogue 3] {datas[random_indies[2]]['dialogue']} [Previous state 3] 'prev_state': {datas[random_indies[2]]['prev_state']} \n [Dialogue 4] {datas[random_indies[3]]['dialogue']} [Previous state 4] 'prev_state': {datas[random_indies[3]]['prev_state']} \n ### Output: [Dialogue 5] {target_dialogue} [Previous State 5] """

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.9)
    # print(f"Prompt:\n{sample['response']}\n")
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    augmented_states[i] = output
    print(
        f"Generated {(i+1)}-th output:\n{output}")
    # print(f"Ground truth:\n{sample['instruction']}")

with open('./prev_states_augment.json', 'w', encoding='utf-8') as fw:
    json.dump(augmented_states, fw, indent="\t", ensure_ascii=False)
