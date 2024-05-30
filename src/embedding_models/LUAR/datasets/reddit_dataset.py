# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import torch

from embedding_models.LUAR.datasets.retrieval_dataset import RetrievalDataset
from embedding_models.LUAR.utilities.file_utils import Utils as utils

class RedditDataset(RetrievalDataset):
    """Torch Dataset object for the Reddit datasets
    """
    def __init__(
        self, 
        params: argparse.Namespace, 
        split: str, 
        num_sample_per_author: int, 
        is_queries=True
    ):
        super().__init__(params, split, num_sample_per_author, is_queries)

        # There are two Reddit datasets available, each with their own files:
        dataset_files = {
            "iur_dataset": {
                "train": ("train.jsonl",),
                "validation": ("train.jsonl", "validation.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            },
            "raw_all": {
                "train": ("data.jsonl",),
                "validation": ("validation_queries.jsonl", "validation_targets.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            },
            "crossnews_gold": {
                "train": ("train.jsonl",),
                "validation": ("train.jsonl", "test_targets.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            },
            "crossnews_silver": {
                "train": ("train.jsonl",),
                "validation": ("train.jsonl", "test_targets.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            },
            "paraphrase_crossnews_silver": {
                "train": ("train.jsonl",),
                "validation": ("train.jsonl", "test_targets.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            },
            "paraphrase_crossnews_gold": {
                "train": ("train.jsonl",),
                "validation": ("train.jsonl", "test_targets.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            },
            "new_crossnews_gold": {
                "train": ("train.jsonl",),
                "validation": ("train.jsonl", "test_targets.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            }
        }

        
        if self.params.dataset_name in dataset_files:
            self.dataset_path = utils.path_exists(os.path.join(utils.data_path, self.dataset_name))
            idx = 0 if is_queries or self.params.sanity else 1
            split = "train" if self.sanity else split
            filename = dataset_files[self.params.dataset_name][split][idx]

            self.filename = os.path.join(self.dataset_path, filename) + self.params.suffix
        else: # NEED TO FIX FOR LATER
            default = {
                "train": ("train.jsonl",),
                "validation": ("train.jsonl", "test_targets.jsonl"),
                "test": ("test_queries.jsonl", "test_targets.jsonl"),
            }
            self.dataset_path = utils.path_exists(os.path.join(utils.data_path, self.dataset_name))
            idx = 0 if is_queries or self.params.sanity else 1
            split = "train" if self.sanity else split
            filename = default[split][idx]

            self.filename = os.path.join(self.dataset_path, filename) + self.params.suffix
            
            # if is_queries:
            #     ending = 'test_queries.jsonl'
            # else:
            #     if split == 'validation':
            #         ending = 'val_targets.jsonl'
            #     elif split == 'test':
            #         ending = 'test_targets.jsonl'
            # self.filename = os.path.join(self.params.dataset_name, ending)
        
        self.load_data(self.filename)
        self.is_test = split != "train"
        
    def __getitem__(
        self, 
        index: int
    ):
        self.fhandle = open(self.filename, "r")

        text = []
        for _ in range(self.num_sample_per_author):
            episode = self.sample_random_episode(index, is_test=self.is_test)
            text.extend(episode["syms"])
                
        if self.split == "train":
            author = torch.tensor([index for _ in range(self.num_sample_per_author)])
        else:
            author = torch.tensor([episode["author_id"] for _ in range(self.num_sample_per_author)])

        data = self.tokenize_text(text)
        if self.params.use_random_windows:
            data = self.sample_random_window(data)
        data = [d.reshape(self.num_sample_per_author, -1, self.params.token_max_length) for d in data]

        self.mask_data_bpe(data)

        return data, author
    
    def get_ids(self, index: int):
        
        self.fhandle = open(self.filename, "r")
        author_data = self.read_line(self.fhandle, index)
        return author_data['document_ids']
