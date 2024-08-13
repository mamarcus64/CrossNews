import pandas as pd
from tqdm import tqdm
import transformers
import torch
import os
from pathlib import Path
import sys
import json
import pdb
import random
import copy
from ast import literal_eval

from attribution_models.attribution_model import AttributionModel

class LLM_Prompting_AA(AttributionModel):
    
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
        return f'prompting_aa_{self.prompt_type}' if '70B' not in self.llm_model_type else f'prompting_aa_{self.prompt_type}_big'
    
    def train_internal(self, params):
        pass
    
    def save_model(self, folder):
        pass
                
    def load_model(self, folder):
        pass
    
    
    def create_prompt(self, query, examples):
        query = ' '.join(query.replace('\n', ' ').split(' ')[:300])
        examples = {k: [' '.join(t.replace('\n', ' ').split(' ')[:80]) for t in v] for k, v in examples.items()}
        authors = [str(x) for x in list(examples.keys())]
        # instruction = "Given a query text and a list of authors with documents written by them, determine the most likely author of the query text. Respond ONLY with the list of author names, from most likely to least likely. For example, if the order of most likelihood was Author 2, then Author 55, then Author 10, respond ONLY with 'Author 2, Author 55, Author 10'."
        instruction = f"Given a query text and a list of authors with documents written by them, determine the most likely authors of the query text. The list of all possible authors is: [{','.join(authors)}]. Respond ONLY with this list, in order from most likely author to least likely author. You will base your answer by comparing the query text to the following provided texts by the authors."
        if self.prompt_type == "task_only":
            instruction = instruction
        elif self.prompt_type == "prompt_av":
            instruction = f"{instruction} Here are some relevant variables to this problem.\n1. punctuation style\n2. special characters\n3. acronyms and abbreviations\n4. writing style\n5.expressions and idioms\n6. tone and mood\n7. sentence structure\n8. any other relevant aspect\nUnderstand the problem, extracting relevant variables and devise a plan to solve the problem. Then carry out the plan."
        elif self.prompt_type == "lip":
            instruction = f"{instruction} Analyze the writing styles of the input texts, disregarding the differences in topic and content. Reason based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings."
        prompt = f"{instruction}\n\nQuery text: {query}\n\n"
        
        for author, texts in examples.items():
            x = '\n\n' # to avoid backslash issue in f-strings
            y = '\n'.join(texts)
            prompt = f'{prompt}TEXTS FROM AUTHOR {author}:{x}{y}{x}'
        
        prompt = f"{prompt}Respond ONLY with the sorted list of potential authors, from most likely to least likely. Respond ONLY in the format, '[{','.join(authors)}]'. Do not include anything else in your response."
        with open('test.txt', 'w', encoding='utf-8') as out:
            out.write(prompt)
        return prompt
            
    def prompt_model(self, query, examples):
        prompt = self.create_prompt(query, examples)
        message = [{"role": "user", "content": prompt}]
        output = self.llm_model(
            message,
            max_new_tokens=512,
            eos_token_id=self.terminators,
            pad_token_id=self.pad_token,
            do_sample=True,
            temperature=0.01,
        )
        result = output[0]['generated_text'][-1]['content']
        return result
    
    
    def evaluate_internal(self, query_df, target_df, df_name=None):
        
        author_ids = sorted([int(x) for x in list(set(query_df['author']))])
        examples = {}
                
        for author in author_ids:
            docs = list(query_df[query_df['author'] == author]['text'])
            if len(docs) < 3:
                examples[author] = docs
            else:
                examples[author] = random.sample(docs, 3)
            
        all_scores = []
        responses = []
        
        valid_responses = 0
        for i in tqdm(list(range(len(target_df)))):
            row = target_df.iloc[i]
            query = row['text']
            
            response = self.prompt_model(query, examples)
            responses.append(response)
            
            try:
                prediction_list = literal_eval(response)
                prediction_list = [int(x) for x in prediction_list]
                
                scores = [-999 + random.random()] * len(author_ids)
                for i, prediction in enumerate(prediction_list):
                    if 0 <= prediction < len(scores):
                        scores[prediction] = -i
                    else:
                        print(f'bad prediction index: {prediction_list}')
                valid_responses += 1
            except:
                scores = [-999 + random.random()] * len(author_ids)
            
            all_scores.append(scores)
            
        #     authors_per_prompt = 4
            
        #     subsample = random.sample(author_ids, len(author_ids))
            
        #     while len(subsample) > 1:
        #         most_likelies = []
                
                
        #         for i in range(0, len(subsample), authors_per_prompt):
        #             prompt_authors = subsample[i:i+authors_per_prompt]
        #             if len(prompt_authors) == 1:
        #                 most_likelies.append(i)
        #                 continue
        #             response = self.prompt_model(query, {k: v for k, v in examples.items() if k in prompt_authors})
        #             most_likelies.append(int(response))
        #         print(len(subsample), len(most_likelies))
        #         subsample = random.sample(most_likelies, len(most_likelies))
            
        #     score = [random.random() * 0.001 for _ in author_ids] # small random noise so there isn't a tie in sorting
        #     score[subsample[0]] += 1
        #     all_scores.append(score)
        print(f'Valid Responses: {valid_responses} / {len(all_scores)}')
            
        return all_scores, responses
