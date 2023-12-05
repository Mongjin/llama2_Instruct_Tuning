import json
from tqdm import tqdm
from transformers import AutoTokenizer
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import nltk
model_id = "/workspace/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def get_data(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for data in fr.readlines():
            datas.append(json.loads(data)['dialogue'])
    return datas


def get_diversity(datas):
    stop_words = set(stopwords.words('english'))
    token_dict = {}
    for data in datas:
        tokens = tokenizer.tokenize(data)
        pos_tags = nltk.pos_tag(tokens)
        for token, pos_tag in zip(tokens, pos_tags):
            if "▁" in token:
                token = token.replace("▁", "")
            if token in stop_words:
                continue
            if not pos_tag.startswith("V"):
                continue
            if token not in token_dict:
                token_dict[token] = 1
            else:
                token_dict[token] += 1
    token_dict = sorted(token_dict.items(), reverse=True, key=lambda item: item[1])
    return token_dict


if __name__ == "__main__":
    datas = get_data('./augmented_dial_gpt-4.jsonl')
    token_dict = get_diversity(datas)
    print(token_dict[:50])
