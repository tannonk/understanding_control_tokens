from muss.feature_extraction import (get_lexical_complexity_score, get_levenshtein_similarity,
                                       get_dependency_tree_depth)

def get_features(src_text, tgt_text):
    return {
        'len_ratio': len(tgt_text) / len(src_text),
        'lev_sim': get_levenshtein_similarity(src_text, tgt_text),
        'word_rank': get_lexical_complexity_score(tgt_text) / get_lexical_complexity_score(src_text),
        'tree_depth': get_dependency_tree_depth(tgt_text) / get_dependency_tree_depth(src_text),
    }