# based on Baichuan2 code

import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments
from utils.prompter import Finetune_Prompter


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)
    num_train_epochs: float = field(default=1.0)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    adam_beta2: float = field(default=0.96)
    lr_scheduler_type: str = field(default="cosine")
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    load_in_8bit: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_target: str = field(default=None)
    conversation_user: str = field(default="user")
    prompt_template: str = field(default="default")
    pad_token: str = field(default=None)
    bos_token: str = field(default=None)
    eos_token: str = field(default=None)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        conversation_user, 
        prompt_template, 
        **kwargs
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.conversation_user = conversation_user
        self.prompter = Finetune_Prompter(prompt_template)
        self.ignore_index = -100
        q_tokens = self.prompter.get_parameter("q_tokens")
        a_tokens = self.prompter.get_parameter("a_tokens")
        qe_tokens = self.prompter.get_parameter("qe_tokens")
        ae_tokens = self.prompter.get_parameter("ae_tokens")
        self.q_str = self.prompter.get_parameter("q_str")
        self.a_str = self.prompter.get_parameter("a_str")
        self.qe_str = self.prompter.get_parameter("qe_str")
        self.ae_str = self.prompter.get_parameter("ae_str")
        self.q_tokens = tokenizer.encode(self.q_str, add_special_tokens=False) if self.q_str and len(self.q_str) else q_tokens
        self.a_tokens = tokenizer.encode(self.a_str, add_special_tokens=False) if self.a_str and len(self.a_str) else a_tokens
        self.qe_tokens = tokenizer.encode(self.qe_str, add_special_tokens=False) if self.qe_str and len(self.qe_str) else qe_tokens
        self.ae_tokens = tokenizer.encode(self.ae_str, add_special_tokens=False) if self.ae_str and len(self.ae_str) else ae_tokens
        self.ret_tokens = tokenizer.encode('\n', add_special_tokens=False)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id != None else 0
        self.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id != None else 0
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id != None else 0
        print("q_tokens", self.q_tokens, "a_tokens", self.a_tokens, "qe_tokens", self.qe_tokens, "ae_tokens", self.ae_tokens, "ret_tokens", self.ret_tokens)
        print("pad_token_id", self.pad_token_id, "bos_token_id", self.bos_token_id, "eos_token_id", self.eos_token_id)

        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = []
        labels = []

        for message in example["conversations"]:
            from_ = message["from"]
            value = message["value"]
            value_ids = self.tokenizer.encode(value, add_special_tokens=False)

            def tokenize(string):
                return self.tokenizer.encode(string, add_special_tokens=False)

            if from_ == self.conversation_user:
                new_ids = self.prompter.generate_prompt("user_input", value, value_ids, self.q_str, self.a_str, self.qe_str, self.ae_str, self.q_tokens, self.a_tokens, self.qe_tokens, self.ae_tokens, self.ret_tokens, self.pad_token_id, self.bos_token_id, self.eos_token_id, tokenize)
                input_ids += new_ids
                labels += [self.ignore_index] * len(new_ids)
                #input_ids += self.q_tokens + value_ids + self.ret_tokens + [self.eos_token_id]
                #labels += [self.ignore_index] * (len(self.q_tokens) + len(value_ids) + len(self.ret_tokens) + 1)
            else:
                new_ids = self.prompter.generate_prompt("bot_input", value, value_ids, self.q_str, self.a_str, self.qe_str, self.ae_str, self.q_tokens, self.a_tokens, self.qe_tokens, self.ae_tokens, self.ret_tokens, self.pad_token_id, self.bos_token_id, self.eos_token_id, tokenize)
                input_ids += new_ids
                labels += new_ids
                #input_ids += self.a_tokens + value_ids + self.ret_tokens + [self.eos_token_id]
                #labels += self.a_tokens + value_ids + self.ret_tokens + [self.eos_token_id]

        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]

        input_ids += [self.pad_token_id] * (
            self.model_max_length - len(input_ids)
        )
        labels += [self.ignore_index] * (self.model_max_length - len(labels))

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.pad_token_id)
        attention_mask = [int(b) for b in attention_mask]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        load_in_8bit=training_args.load_in_8bit, 
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        pad_token=training_args.pad_token,
        bos_token=training_args.bos_token,
        eos_token=training_args.eos_token,
    )

    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = 0

    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

        linear_module_names = set()
        for name, module in model.named_modules():
            mtype = type(module).__name__
            if "Linear" in mtype:
                names = name.split('.')
                linear_module_names.add(names[-1] + ":" + mtype)

        for name in linear_module_names:
            print(name)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=training_args.lora_target.split(","),
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, **vars(training_args) #training_args.conversation_user, training_args.prompt_template
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
