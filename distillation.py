import fire
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def main(
    base_model: str = "",
    prompt_template: str = "",
    ):

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(base_model)

    with open('../alpaca_oaast_sharegpt_cot_firefly_belle35_80w.json', 'r') as f:
        json_data = json.load(f)

    for data in json_data:
        if len(data['conversations']) <= 2:
            continue
        messages = []
        for conv in data['conversations'][:-1]:
            messages.append({'role':conv['from'], 'content':conv['value']})

        print("=================")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(prompt)
        inputs = tokenizer([prompt], return_tensors="pt")
        #inputs = inputs.to('cpu')
        pred = model.generate(**inputs, generation_config=generation_config, max_new_tokens=3000, do_sample=False)
        output = tokenizer.decode(pred.cpu()[0])
        print("-----------------")
        print(output)
        last = data['conversations'][-1]
        messages.append({'role':last['from'], 'content':output[len(prompt):]})
        print(messages)
        break

fire.Fire(main)
