import fire
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from utils.prompter import Prompter

def main(
    base_model: str = "",
    prompt_template: str = "",
    ):

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(base_model)

    df = pd.read_excel('batch_infer/questions.xlsx')

    for index, row in df.iterrows():
        print("=================")
        print(row['q'])
        prompter = Prompter(prompt_template)
        prompt = prompter.generate_prompt(row['q'])
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, generation_config=generation_config, max_new_tokens=1000, do_sample=False)
        output = tokenizer.decode(pred.cpu()[0])
        output = prompter.get_response(output)
        df.loc[index, 'a'] = output
        print(output)

    base_model_parts = base_model.split("/")
    df.to_excel('batch_infer/output_{}.xlsx'.format(base_model_parts[-1] if len(base_model_parts[-1]) else base_model_parts[-2]), index=False)

fire.Fire(main)
