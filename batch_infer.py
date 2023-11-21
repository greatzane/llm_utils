import fire
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from utils.prompter import Prompter

def main(
    base_model: str = "",
    prompt_template: str = "",
    eos_token: str = None
    ):

    tokenizer = AutoTokenizer.from_pretrained(base_model, eos_token=eos_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(base_model)
    print("eos_token_id", tokenizer.eos_token_id)

    df = pd.read_csv('batch_infer/questions.csv', encoding="utf-8")

    for index, row in df.iterrows():
        print("=================")
        print(row['q'])
        prompter = Prompter(prompt_template)
        prompt = prompter.generate_prompt(row['q'])
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, generation_config=generation_config, max_new_tokens=1000, do_sample=False, eos_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(pred.cpu()[0])
        output = prompter.get_response(output)
        df.loc[index, 'a'] = output
        print("-----------------")
        print(output)

    base_model_parts = base_model.split("/")
    df.to_csv('batch_infer/output_{}.csv'.format(base_model_parts[-1] if len(base_model_parts[-1]) else base_model_parts[-2]), index=False, encoding="utf-8")

fire.Fire(main)
