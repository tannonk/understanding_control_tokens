#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Tannon Kew (kew@cl.uzh.ch)

from string import Template
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
from torch import Tensor
from tqdm import tqdm
from itertools import combinations
import re
# import warnings

ctrl1 = Template('<DEPENDENCYTREEDEPTHRATIO_${tree_depth}>')
ctrl2 = Template('<WORDRANKRATIO_${word_rank}>')
ctrl3 = Template('<REPLACEONLYLEVENSHTEIN_${lev_sim}>')
ctrl4 = Template('<LENGTHRATIO_${len_ratio}>')

def read_lines(filepath):
    lines = []
    with open(filepath, 'r', encoding='utf8') as inf:
        # assumed infile contains lines with either src or src[tab]tgt1[tab]tgt2...
        for line in inf:
            line = line.split('\t')[0]
            lines.append(line.strip())
    return lines

def construct_input_for_access(src: str, params: Dict) -> str:
    '''build input string with ctrl token params for ACCESS/MUSS model'''
    input_string = ctrl1.substitute(tree_depth=params['tree_depth']) + ' ' \
                    + ctrl2.substitute(word_rank=params['word_rank']) + ' ' \
                    + ctrl3.substitute(lev_sim=params['lev_sim']) + ' ' \
                    + ctrl4.substitute(len_ratio=params['len_ratio']) + ' ' \
                    + src
    
    return input_string

def check_uniqueness(batch: Tensor) -> None:
    '''compares all encoded tensors in a batch'''
    for a, b in combinations(batch, 2):
        if all(torch.eq(a, b)):
            raise warnings.warn('Tensors in batch are expected to be different but found equality.')
    return

def batch(iterable: List[str], n: int = 5):
    '''batching generator'''
    l = len(iterable)
    for indx in range(0, l, n):
        yield iterable[indx:min(indx + n, l)]

def generate(sentences: List[str], model, tokenizer):
    '''run generation with a hugging face model'''
    input_ids = tokenizer(sentences, padding=True, return_tensors='pt').to(model.device)
    check_uniqueness(input_ids['input_ids'])
    outputs_ids = model.generate(
        input_ids["input_ids"], 
        num_beams=5, 
        max_length=128)
    outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outputs
    
# def len_norm_avg_edit_distance(refs: List[str], verbose: bool = False, tokenize: bool = False) -> Tuple[float]:
#     """
#     Computes length-normalised average edit distance for a set of similiar sentences
    
#     This is an adaption of the intrinsic uncertainty measure for multi-reference sets proposed by Stahlberg et al., 2022
#     """

#     if tokenize:
#         refs = [' '.join(tokenizer.tokenize(ref)) for ref in refs]
    
#     # average ref length
#     n = len(refs)

#     tot_lens = 0
#     for r in refs:
#         tot_lens += len(r.split())
#     if tot_lens == 0:
#         return None, None, None
#     avg_ref_len = (1/n) * tot_lens
    
#     # avg edit dist between refs
#     tot_dist = 0
#     for r1, r2 in combinations(refs, 2):
#         d = edit_distance(r1.split(), r2.split(), substitution_cost=1)
#         tot_dist += d
#     avg_edit_dist = tot_dist / ((n * (n-1) / 2))

#     if verbose:
#         print('tot_lens:', tot_lens, '\ttot_dist:', tot_dist)
#         print('avg_lens:', avg_ref_len, '\tavg_dist:', avg_edit_dist)

#     return avg_edit_dist / avg_ref_len



def encode_params(params, range_min, range_max, step_size):
    values = list(np.arange(range_min, range_max, step_size))
    
    len_ratio_enc = np.zeros(len(values))
    lev_sim_enc = np.zeros(len(values))
    word_rank_enc = np.zeros(len(values))
    tree_depth_enc = np.zeros(len(values))
    
    len_ratio_enc[values.index(params['len_ratio'])] = 1
    lev_sim_enc[values.index(params['lev_sim'])] = 1
    word_rank_enc[values.index(params['word_rank'])] = 1
    tree_depth_enc[values.index(params['tree_depth'])] = 1

    return np.concatenate((len_ratio_enc, lev_sim_enc, word_rank_enc, tree_depth_enc))

def construct_single_label_dataset(
    sentences: List[str], 
    ctrl_token: str = 'len_ratio', 
    range_min: float = 0.25, 
    range_max: float = 1.5, 
    step_size: float = 0.25,
    ):

    inputs = []
    labels = []

    for sent in tqdm(sentences, total=len(sentences)):
        # possible combinations = num_trials^4
        for a in np.arange(range_min, range_max, step_size):
            a = round(a, 2) # target values = 0.0, 0.05, 0.1, 0.15, ...
            params = {
                'len_ratio': a if ctrl_token == 'len_ratio' else 1.0,
                'lev_sim': a if ctrl_token == 'lev_sim' else 1.0,
                'word_rank': a if ctrl_token == 'word_rank' else 1.0,
                'tree_depth': a if ctrl_token == 'tree_depth' else 1.0,
                }

            inputs.append(construct_input_for_access(sent, params))
            labels.append([str(a)])      

    assert len(inputs) == len(labels)

    print(f'*** Constructed {len(inputs)} inputs ***')
    
    return inputs, labels

def construct_multi_label_dataset(
    sentences: List[str],
    range_min: float = 0.25, 
    range_max: float = 1.5, 
    step_size: float = 0.25, 
    ):
    
    inputs = []
    labels = []
    
    for sent in tqdm(sentences, total=len(sentences)):

        # possible combinations = num_trials^4
        for a in np.arange(range_min, range_max, step_size):
            a = round(a, 2) # target values = 0.0, 0.05, 0.1, 0.15, ...
            for b in np.arange(range_min, range_max, step_size):
                b = round(b, 2) # target values = 0.0, 0.05, 0.1, 0.15, ...
                for c in np.arange(range_min, range_max, step_size):
                    c = round(c, 2) # target values = 0.0, 0.05, 0.1, 0.15, ...
                    for d in np.arange(range_min, range_max, step_size):
                        d = round(d, 2) # target values = 0.0, 0.05, 0.1, 0.15, ...
                        
                        params = {
                            'len_ratio': a,
                            'lev_sim': b,
                            'word_rank': c,
                            'tree_depth': d,
                        }
                        
                        inputs.append(construct_input_for_access(sent, params))
                        labels.append(encode_params(params, range_min, range_max, step_size))

    assert len(inputs) == len(labels)
    
    print(f'*** Constructed {len(inputs)} inputs ***')

    return inputs, labels

if __name__ == "__main__":
    sentences = read_lines('/scratch/tkew/ctrl_tokens/resources/data/examples.en')
    sentences, labels = construct_multi_label_dataset(sentences, 0.25, 1.5, 0.25) # results in 625 examples per sentence!
    for s in sentences[:625]:
        print(s)