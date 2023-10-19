"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        if len(self.template["response_split"]) > 0:
            return output.split(self.template["response_split"])[1].strip()
        else:
            return output

class Finetune_Prompter(object):
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "default"
        file_name = osp.join("templates", f"finetune_{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using finetune prompt template finetune_{template_name}"
            )

    def get_parameter(self, key):
        if key in self.template:
            return self.template[key]

    def generate_prompt(
        self,
        key,
        input_ids,
        q_tokens, 
        a_tokens, 
        qe_tokens, 
        ae_tokens, 
        ret_tokens, 
        pad_token_id, 
        bos_token_id, 
        eos_token_id
    ) -> []:

        ret = []
        for item in self.template[key]:
            if item == "input" and input_ids and len(input_ids):
                ret += input_ids
            elif item == "q_tokens" and q_tokens and len(q_tokens):
                ret += q_tokens
            elif item == "a_tokens" and a_tokens and len(a_tokens):
                ret += a_tokens
            elif item == "qe_tokens" and qe_tokens and len(qe_tokens):
                ret += qe_tokens
            elif item == "ae_tokens" and ae_tokens and len(ae_tokens):
                ret += ae_tokens
            elif item == "ret_tokens" and ret_tokens and len(ret_tokens):
                ret += ret_tokens
            elif item == "pad_token":
                ret += [pad_token_id]
            elif item == "bos_token":
                ret += [bos_token_id]
            elif item == "eos_token":
                ret += [eos_token_id]
        
        return ret
