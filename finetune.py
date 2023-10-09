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
    bf16: float = field(default=True)
    tf32: float = field(default=True)
    use_lora: bool = field(default=False)
    lora_target: str = field(default=None)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100
        self.system_title = tokenizer('<<<system>>>\n').input_ids
        self.conversations_title = tokenizer('\n\n<<<conversations>>>\n').input_ids
        self.q_tokens = tokenizer('<Q>:').input_ids
        self.a_tokens = tokenizer('<A>:').input_ids
        self.ret_token = tokenizer('\n').input_ids
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id != None else 0
        self.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id != None else 0
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id != None else 0

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

        if "system" in example:
            system_ids = self.tokenizer.encode(example["system"])
            input_ids += self.system_title + system_ids + self.conversations_title
            labels += [self.ignore_index] * (len(self.system_title) + len(system_ids) + len(self.conversations_title))

        for message in example["conversations"]:
            from_ = message["from"]
            value = message["value"]
            value_ids = self.tokenizer.encode(value)

            if from_ == "human":
                input_ids += self.q_tokens + value_ids + self.ret_token
                labels += [self.ignore_index] * (len(self.q_tokens) + len(value_ids) + len(self.ret_token))
            else:
                input_ids += self.a_tokens + value_ids + self.ret_token
                labels += self.a_tokens + value_ids + self.ret_token

        input_ids.append(self.eos_token_id)
        labels.append(self.eos_token_id)
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]

        input_ids += [self.pad_token_id] * (
            self.model_max_length - len(input_ids)
        )
        labels += [self.ignore_index] * (self.model_max_length - len(labels))

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.pad_token_id)
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
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )

    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = 0

    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=training_args.lora_target,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
