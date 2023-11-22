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
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = self.template["prompt"].format(
            instruction=instruction
        )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        if len(self.template["response_split"]) > 0:
            output = output.split(self.template["response_split"])[1].strip()
            if "response_end_split" in self.template and len(self.template["response_end_split"]) > 0:
                output = output.split(self.template["response_end_split"])[0].strip()
            return output
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
        input_str, 
        input_ids,
        q_str, 
        a_str, 
        qe_str, 
        ae_str, 
        q_tokens, 
        a_tokens, 
        qe_tokens, 
        ae_tokens, 
        ret_tokens, 
        pad_str,
        bos_str,
        eos_str,
        pad_token_id, 
        bos_token_id, 
        eos_token_id, 
        process_string, 
    ) -> []:

        ret = []
        temp_str = ""
        for item in self.template[key]:
            if not item.endswith("_str") and len(temp_str):
                ret += process_string(temp_str)
                temp_str = ""

            if item == "input_str" and input_str and len(input_str):
                temp_str += input_str
            elif item == "q_str" and q_str and len(q_str):
                temp_str += q_str
            elif item == "a_str" and a_str and len(a_str):
                temp_str += a_str
            elif item == "qe_str" and qe_str and len(qe_str):
                temp_str += qe_str
            elif item == "ae_str" and ae_str and len(ae_str):
                temp_str += ae_str
            elif item == "ret_str":
                temp_str += "\n"
            elif item == "pad_str":
                temp_str += pad_str
            elif item == "bos_str":
                temp_str += bos_str
            elif item == "eos_str":
                temp_str += eos_str
            elif item == "input_tokens" and input_ids and len(input_ids):
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
        
        if len(temp_str):
            ret += process_string(temp_str)

        return ret
