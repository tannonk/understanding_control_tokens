
"""

python construct_inference_file_with_mixed_control_tokens.py \
    --dataset resources/data/examples.en \
    --outpath resources/data/en/inference \
    --param_dist id --sample --sample_size 0.1 --seed 4

python construct_inference_file_with_mixed_control_tokens.py \
    --dataset resources/data/en/aligned/asset_test.tsv \
    --outpath resources/data/en/inference \
    --param_dist id --sample --sample_size 0.1 --seed 4

"""

import math
import random
import argparse
# from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Union
from tqdm import tqdm
import pandas as pd

from muss_utils import (
    compute_features, 
    fetch_preprocessor_used_in_muss_model_training, 
    build_processor_combinations, 
    construct_multiple_training_instances,
    get_params_as_dict,
    strip_params,
    build_processor_values,
)

from utils import read_lines

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='resources/data/en/aligned/asset_test.tsv')
    parser.add_argument('--outpath', type=Path, default='resources/data/en/inference/')
    parser.add_argument('--param_dist', type=str, choices=['id', 'ood'], default='id')
    parser.add_argument('--sample_size', type=float, default=0.1)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--seed', type=int, default=4)
    return parser.parse_args()

# dataset = 'examples.en' # 
# dataset = 'asset_test' # 320

# if dataset.startswith('asset'):
#     file = f'resources/data/en/aligned/{dataset}.tsv'    
# # elif dataset.startswith('turk'):
# #     file = f'/scratch/tkew/ctrl_tokens/resources/data/en/aligned/{dataset}.tsv'    
# elif dataset == 'examples.en':
#     file = f'resources/data/{dataset}'    

def construct_inference_file(dataset: str, outfile: str, param_dist: str, sample_size: float = 0.1, seed: int = 4):
    """
    Generate inference file for MUSS model.
    """
    
    preprocessors = fetch_preprocessor_used_in_muss_model_training()

    param_combinations = build_processor_combinations(preprocessors, min_max_step_vals)

    
    if sample_size == 0.0: # will generate all possible src sentences (this is a lot!)
        lc = 0
        with open(outfile, 'w', encoding='utf8') as outf:
            for line in tqdm(read_lines(dataset)):
                for src_sent in construct_multiple_training_instances(line, param_combinations):
                    lc += 1
                    outf.write(src_sent + '\n')
        
    else:
        src_sents = []
        for line in tqdm(read_lines(dataset)):
            for src_sent in construct_multiple_training_instances(line, param_combinations):
                src_sents.append(src_sent)

        # print(len(src_sents))
        # can sample given a percentage of the total number of output sentences    
        
        actual_sample_size = int(len(src_sents)*float(sample_size))
        random.seed(seed)
        src_sents = random.sample(src_sents, actual_sample_size)
            
        lc = 0
        with open(outfile, 'w', encoding='utf8') as outf:
            for src_sent in src_sents:
                lc += 1
                outf.write(src_sent + '\n')
        
    print(f'Constructed {lc} sentences with ACCESS/MUSS control tokens in {outfile}')        

    return

if __name__ == "__main__":
    args = set_args()
    
    Path(args.outpath).mkdir(parents=True, exist_ok=True)
    outfile = args.outpath / f'{args.dataset.stem}_{args.sample_size}_{args.seed}_{args.param_dist}.txt'

    # set based on distribution of values in training data
    if args.param_dist == 'id':
        
        min_max_step_vals = {
            'LENGTHRATIO': (0.7, 1.2, 0.05), 
            'REPLACEONLYLEVENSHTEIN': (0.5, 1.0, 0.05), 
            'WORDRANKRATIO': (0.7, 1.2, 0.05), 
            'DEPENDENCYTREEDEPTHRATIO': (0.5, 1.5, 0.05),
        }
    
    elif args.param_dist == 'ood':
        
        min_max_step_vals = {
            'LENGTHRATIO': (0.25, 1.26, 0.25),
            'REPLACEONLYLEVENSHTEIN': (0.25, 1.26, 0.25),
            'WORDRANKRATIO': (0.25, 1.26, 0.25),
            'DEPENDENCYTREEDEPTHRATIO': (0.25, 1.26, 0.25),
        }

    construct_inference_file(
        dataset=args.dataset, outfile=outfile, 
        param_dist=args.param_dist, sample_size=args.sample_size, seed=args.seed
        )
    
    print('done')