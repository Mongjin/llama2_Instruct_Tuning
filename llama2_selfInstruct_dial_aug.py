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

tokenizer = AutoTokenizer.from_pretrained(model_id)

datas = []
with open('samples_translation.json', 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        datas.append(json.loads(line))

augmented_dialogues = {}
for i in range(0, 15):
    random_indies = []
    while len(random_indies) < 6:
        index = random.randint(0, 59)
        if index not in random_indies:
            random_indies.append(index)

    prompt = f"""### Instruction: Please generate new dialogue that 'user' is asking recommendation food or travel for 'bot'. 'bot' should respond like dialogue agent that request more information for better recommendation rather than recommend directly. You should follow the structure of given samples; You should always start with user's utterance and finish wih "bot: ".; Lastly, please try to generate new dialogue that is not similiar with samples. \n ### Input: [Sample 1] {datas[random_indies[0]]['dialogue']} \n [Sample 2] {datas[random_indies[1]]['dialogue']} \n [Sample 3] {datas[random_indies[2]]['dialogue']} \n [Sample 4] {datas[random_indies[3]]['dialogue']} \n [Sample 5] {datas[random_indies[4]]['dialogue']} \n [Sample 6] {datas[random_indies[5]]['dialogue']} \n ### Output: [Sample 7] """

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.9)
    # print(f"Prompt:\n{sample['response']}\n")
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    augmented_dialogues[i] = output
    print(
        f"Generated {(i+1)}-th instruction:\n{output}")
    # print(f"Ground truth:\n{sample['instruction']}")

with open('./Dialogue_augment.json', 'w', encoding='utf-8') as fw:
    json.dump(augmented_dialogues, fw, indent="\t", ensure_ascii=False)
