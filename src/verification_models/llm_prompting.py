import pandas as pd
from tqdm import tqdm
import transformers
import torch
import os
from pathlib import Path
import sys
import json

from verification_models.verification_model import VerificationModel

class LLM_Prompting(VerificationModel):
    
    def __init__(self, args, parameter_set):
        self.llm_model_type = parameter_set['llm_model']
        self.prompt_type = parameter_set['prompt']
        super().__init__(args, parameter_set)
        
        self.llm_model = transformers.pipeline(
            "text-generation",
            model=self.llm_model_type,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        self.terminators = [
            self.llm_model.tokenizer.eos_token_id,
            self.llm_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        self.pad_token = self.llm_model.tokenizer.eos_token_id
        
    
    def get_model_name(self):
        return f'prompting_{self.prompt_type}' if '70B' not in self.llm_model_type else f'prompting_{self.prompt_type}_big'
    
    def train_internal(self, params):
        pass
    
    def save_model(self, folder):
        pass
                
    def load_model(self, folder):
        pass
    
    
    def create_prompt(self, text0, text1):
        instruction = "Verify if two input texts were written by the same author. Provide your answer simply with True or False."
        if self.prompt_type == "task_only":
            instruction = instruction
        elif self.prompt_type == "prompt_av":
            instruction = f"{instruction} Here are some relevant variables to this problem.\n1. punctuation style\n2. special characters\n3. acronyms and abbreviations\n4. writing style\n5.expressions and idioms\n6. tone and mood\n7. sentence structure\n8. any other relevant aspect\nUnderstand the problem, extracting relevant variables and devise a plan to solve the problem. Then carry out the plan. Remember to provide your answer ONLY with True or False. Do NOT respond with anything other than \"True\" or \"False\"."
        elif self.prompt_type == "lip":
            instruction = f"{instruction} Analyze the writing styles of the input texts, disregarding the differences in topic and content. Reason based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings."
        prompt = f"{instruction}\n\nInput Text 1: {text0}\n\nInput Text 2: {text1}\n\nAnswer:"
        return prompt
            
    def prompt_model(self, text0, text1):
        prompt = self.create_prompt(text0, text1)
        message = [{"role": "user", "content": prompt}]
        output = self.llm_model(
            message,
            max_new_tokens=512,
            eos_token_id=self.terminators,
            pad_token_id=self.pad_token,
            do_sample=True,
            temperature=0.01,
        )
        return output[0]['generated_text'][-1]['content']
    
    def evaluate_internal(self, df, df_name=None):
        
        predictions, labels = [], []
        
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            try:
                text0 = ' '.join(str(row['text0']).split(' ')[:750])
                text1 = ' '.join(str(row['text1']).split(' ')[:750])
            except:
                print(row['text0'])
                print(row['text1'])
            model_response = self.prompt_model(text0, text1)
            
            if 'true' in model_response.lower():
                prediction = 1
            elif 'false' in model_response.lower():
                prediction = 0
            else:
                prediction = 0
                print(f'bad model response: {model_response}')
            
            predictions.append(prediction)
            labels.append(int(row['label']))
        
        return predictions, labels
