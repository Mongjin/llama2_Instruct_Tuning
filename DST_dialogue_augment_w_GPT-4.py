from openai import OpenAI
import json
import random
from tqdm import tqdm

api_key = input("Please enter your API KEY: ")
API_KEY = api_key  ## 본인 API key 입력
ENGINE = 'gpt-4-1106-preview'  ## 지금 버전이 gpt-4 turbo!
client = OpenAI(api_key=API_KEY)


def run_gpt_turbo(engine, prompt):
    completion = client.chat.completions.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],  # 입력 prompt
        max_tokens=2048,
        temperature=0.0,
        top_p=1,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
        # logit_bias=
        # stop=[],
    )

    print(completion)

    answer = completion.choices[0].message.content
    usage = completion.usage

    return answer, usage


def run_text_davinci(engine, prompt, max_tokens, temperature, top_p,
                     frequency_penalty, presence_penalty, logprobs, n, best_of, stop_sequences=None, debug=False
                     ):
    response = None

    try:
        prompt += "\n"
        response = client.completions.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_sequences,
            logprobs=logprobs,
            n=n,
            best_of=best_of)
        if debug:
            return response["choices"][0]["text"], response["usage"]["total_tokens"], response
        return response["choices"][0]["text"], response["usage"]["total_tokens"]

    except Exception as e:
        print(e)
        return None, None


def get_datas(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            datas.append(json.loads(line))
    return datas


seeds = get_datas('./seed_data_v2.jsonl')
augmented_dials = get_datas('./augmented_dial_gpt-4.jsonl')

answers = []
for iter in tqdm(range(60), desc=f"Completing..."):
    rand_seeds_indies = []
    seeds_pool = []
    while len(rand_seeds_indies) < 2:
        index = random.randint(0, len(seeds) - 1)
        if index not in rand_seeds_indies:
            rand_seeds_indies.append(index)
            seeds_pool.append(seeds[index])
    rand_augs_indies = []
    augs_pool = []
    while len(rand_augs_indies) < 1:
        index = random.randint(0, len(augmented_dials) - 1)
        if index not in rand_augs_indies:
            rand_augs_indies.append(index)
            augs_pool.append(augmented_dials[index])

    samples_pool = seeds_pool + augs_pool
    random.shuffle(samples_pool)
    # for i in range(len(seeds)):
    #     data = seeds[i]
    #     target_dialogue = data['dialogue']
    #     try:
    #         bot_index = target_dialogue.rindex("bot:")
    #         data['dialogue'] = target_dialogue[:bot_index].strip()
    #     except:
    #         pass

    # 본인 프롬프트에 맞게 수정
    prompt = f'''### Instruction: Please generate new dialogue that 'user' is asking recommendation food or travel for 'bot'. 'bot' should respond like dialogue agent that request more information for better recommendation rather than recommend directly. You should follow the structure of given samples; Lastly, you should avoid to generate similiar bot's response in samples, Please create diverse bot's responses which is requesting new information to user. \n ### Input: [Sample 1] {samples_pool[0]['dialogue']} \n [Sample 2] {samples_pool[1]['dialogue']} \n [Sample 3] {samples_pool[2]['dialogue']} \n ### Output: [Sample 4] '''

    answer, usage = run_gpt_turbo(ENGINE, prompt=prompt)
    answers.append({'prev_state': "", "dialogue": answer, "cur_state": "", "response": ""})
    augmented_dials.append({'prev_state': "", "dialogue": answer, "cur_state": "", "response": ""})

with open('augmented_dial_v2_gpt-4.jsonl', 'w', encoding='utf-8') as fw:
    for data in answers:
        fw.write(json.dumps(data, ensure_ascii=False))
        fw.write("\n")