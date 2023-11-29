import openai


def run_gpt_turbo(api_key, engine, prompt):
    openai.api_key = api_key
    completion = openai.ChatCompletion.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}], # 입력 prompt
        max_tokens=2048,
        temperature=0.0,
        top_p=1,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
        # logit_bias=
        # stop=[],
    )

    answer = completion['choices'][0]['message']['content'].strip()
    usage = completion["usage"]
    
    return answer, usage

def run_text_davinci(
        api_key, engine, prompt, max_tokens, temperature, top_p, 
        frequency_penalty, presence_penalty, logprobs, n, best_of, stop_sequences=None, debug=False
    ):
    response = None
    openai.api_key = api_key
    
    try:
        prompt += "\n"
        response = openai.Completion.create(
            api_key=api_key,
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
            best_of=best_of,
        )
        if debug:
            return response["choices"][0]["text"], response["usage"]["total_tokens"], response
        return response["choices"][0]["text"], response["usage"]["total_tokens"]
    
    except Exception as e:
        print(e)
        return None, None


API_KEY = '' ## 본인 API key 입력
ENGINE = 'gpt-4.0-turbo'

# prompt = '''### Instruction: Please generate new dialogue that 'user' is asking recommendation food or travel for 'bot'. 'bot' should respond like dialogue agent that request more information for better recommendation rather than recommend directly. You should follow the structure of given samples; You should always start with user's utterance and finish wih "bot: ".; Lastly, you should try to avoid similiar dialogue with samples. \n ### Input: [Sample 1] {datas[random_indies[0]]['dialogue']} \n [Sample 2] {datas[random_indies[1]]['dialogue']} \n [Sample 3] {datas[random_indies[2]]['dialogue']} \n [Sample 4] {datas[random_indies[3]]['dialogue']} \n [Sample 5] {datas[random_indies[4]]['dialogue']} \n [Sample 6] {datas[random_indies[5]]['dialogue']} \n ### Output: [Sample 7] '''
prompt = '''### Instruction: Please generate new dialogue that 'user' is asking recommendation food or travel for 'bot'. 'bot' should respond like dialogue agent that request more information for better recommendation rather than recommend directly. You should follow the structure of given samples; You should always start and finish with user's utterance; Lastly, you should try to avoid similiar dialogue with samples. \n ### Input: [Sample 1] {datas[random_indies[0]]['dialogue']} \n [Sample 2] {datas[random_indies[1]]['dialogue']} \n [Sample 3] {datas[random_indies[2]]['dialogue']} \n [Sample 4] {datas[random_indies[3]]['dialogue']} \n [Sample 5] {datas[random_indies[4]]['dialogue']} \n [Sample 6] {datas[random_indies[5]]['dialogue']} \n ### Output: [Sample 7] '''

answer, usage = run_gpt_turbo(API_KEY, ENGINE, prompt='')