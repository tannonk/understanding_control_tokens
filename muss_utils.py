
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import numpy as np
import itertools

from muss.mining.training import get_bart_kwargs
from muss.preprocessors import get_preprocessors
from muss.feature_extraction import (get_lexical_complexity_score, get_levenshtein_similarity,
                                       get_dependency_tree_depth)


# self implementation according to description in the paper (may not be reliable)
def get_features(src, tgt):
    return {
        'len_ratio': len(tgt) / len(src),
        'lev_sim': get_levenshtein_similarity(src, tgt),
        'word_rank': get_lexical_complexity_score(tgt) / get_lexical_complexity_score(src),
        'tree_depth': get_dependency_tree_depth(tgt) / get_dependency_tree_depth(src),
    }

# # This dataset should exist in resources/datasets/ and contain the following files:
# # train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
def compute_features(src, tgt, preprocessors, as_score=False):
    """
    computes features for a single pair of source and target as done in MUSS model training

    if `as_score` is False, the src and tgt sentences and special token are not returned in output dict
    """
    data = {'src': src, 'tgt': tgt} if not as_score else {}

    for preprocessor in preprocessors:
        try:
            # equivalent to preprocessor.encode_sentence_pair(c, s), but get intermediate results
            feat_val = preprocessor.get_feature_value(src, tgt)
            bucketed_feat_val = preprocessor.bucketize(feat_val)
            special_token = preprocessor.get_feature_token(bucketed_feat_val)
            # preprocessor.get_feature_token(preprocessor.bucketize())
            # print(preprocessor.prefix, preprocessor.bucketize(preprocessor.get_feature_value(c, s)))
            # print(preprocessor.prefix, preprocessor.get_feature_value(c, s))
            data[f'{preprocessor.prefix}_score'] = feat_val
            data[f'{preprocessor.prefix.upper()}'] = bucketed_feat_val
            if not as_score:
                data[f'{preprocessor.prefix}_token'] = special_token
            
        except AttributeError:
            pass
    
    return data


def fetch_preprocessor_used_in_muss_model_training(lang='en'):
    kwargs = get_bart_kwargs(
        dataset=None, language=lang, use_access=True
    )
    preprocessors_kwargs = kwargs.get('preprocessors_kwargs', {})
    
    if 'GPT2BPEPreprocessor' in preprocessors_kwargs:
        # not needed for analyses, so remove it
        preprocessors_kwargs.pop('GPT2BPEPreprocessor')
    
    preprocessors = get_preprocessors(preprocessors_kwargs)
    print('Loaded preprocessors:', preprocessors)

    return preprocessors

# def build_processor_combinations_basic(
#     preprocessors,
#     range_min: float = 0.25, 
#     range_max: float = 1.5, 
#     step_size: float = 0.25, 
#     ):
#     """
#     NOTE: superseded by `build_processor_combinations`
#     NOTE: only need to do this once
#      """

#     # import pdb;pdb.set_trace()
#     # inputs = []
#     # labels = []
#     # possible combinations = num_trials^4
#     assert len(preprocessors) == 4, f"Only 4 preprocessors are supported but found {len(preprocessors)}"

#     param_combinations = []

#     for a in np.arange(range_min, range_max, step_size):
#         a = round(a, 2) # target values = 0.0, 0.05, 0.1, 0.15, ...
#         for b in np.arange(range_min, range_max, step_size):
#             b = round(b, 2) # target values = 0.0, 0.05, 0.1, 0.15, ...
#             for c in np.arange(range_min, range_max, step_size):
#                 c = round(c, 2) # target values = 0.0, 0.05, 0.1, 0.15, ...
#                 for d in np.arange(range_min, range_max, step_size):
#                     d = round(d, 2) # target values = 0.0, 0.05, 0.1, 0.15, ...
                    
#                     params = [
#                         preprocessors[0].get_feature_token(a),
#                         preprocessors[1].get_feature_token(b),
#                         preprocessors[2].get_feature_token(c),
#                         preprocessors[3].get_feature_token(d),
#                     ]

#                     # yield params
#                     param_combinations.append(params)
    
#     return param_combinations

def build_processor_values(
    preprocessor,
    range_min: float = 0.25, 
    range_max: float = 1.5, 
    step_size: float = 0.25, 
    ):

    param_values = []
    for a in np.arange(range_min, range_max, step_size):
        a = round(a, 2)
        param_values.append(preprocessor.get_feature_token(a))
    return param_values


def build_processor_combinations(preprocessors: List, min_max_step_vals: Dict, verbose: bool = False):

    params = []
    # note: model is trained with ctrl tokens in following order: Dep, Word, Lev, Len!
    for i, preprocessor in enumerate(reversed(preprocessors)):
        params.append(build_processor_values(preprocessor, *min_max_step_vals[preprocessor.feature_name]))

    if verbose:
        print('Possible params:')
        print(params)
        print()

    param_combinations = list(itertools.product(*params))
    
    if verbose:
        print('Possible param combinations:')
        print(f'Total number of possible parameter combinations: {len(param_combinations)}')
        print(f'e.g.: {param_combinations[:4]} ...')
        print()

    # IMPORTANT: check all param settings are valid according to order according to how the model is trained!
    for pset in param_combinations:
        assert len(pset) == 4, f'invalid param set: {pset}'
        assert pset[0].startswith('<DEPENDENCYTREEDEPTHRATIO'), f'invalid param set: {pset}'
        assert pset[1].startswith('<WORDRANKRATIO'), f'invalid param set: {pset}'
        assert pset[2].startswith('<REPLACEONLYLEVENSHTEIN'), f'invalid param set: {pset}'
        assert pset[3].startswith('<LENGTHRATIO'), f'invalid param set: {pset}'

    return param_combinations

def construct_multiple_training_instances(src, params):
    for param_list in params:
        yield ' '.join(list(param_list) + [src])
        
def strip_params(src: str, preprocessors: List) -> str:
    """
    returns the original src text
    """
    src = ' '.join(src.split()[len(preprocessors):]).strip()
    return src
    
def get_params_as_dict(src: str, preprocessors: List) -> Dict:
    """returns the parameters from a source text as a dict"""
    
    params = src.split()[:len(preprocessors)]
    params_dict = {}
    for p in params:
        name, value = p.strip('<>').split('_')
        params_dict[name] = float(value)
    return params_dict


if __name__ == '__main__':

    # import pdb;pdb.set_trace()
    dataset = '/scratch/tkew/muss/resources/datasets/muss_mined_paraphrases/en_mined_paraphrases/'

    preprocessors = fetch_preprocessor_used_in_muss_model_training()

    param_combinations = build_processor_combinations(preprocessors)

    with open(Path(dataset) / 'train_head200.csv', 'r', encoding='utf8') as f:
        for line in tqdm(f):
            src, tgt = line.strip().split('\t')
            
            # compute of features on sentence pairs
            # data = compute_features(src, tgt, preprocessors)
            
            # reconstruct multiple training instances from a single src sentence
            # for processor_combination in build_processor_combinations(preprocessors, 0.5, 1.0, 0.1):
            #     print(i)
            for s in construct_multiple_training_instances(src, param_combinations):
                print(s)
            

    print('done')