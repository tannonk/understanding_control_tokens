#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

PROOF OF CONCEPT

AIM: get all hidden states from pretrained BART model and save them to a file

ISSUES: currently, batching is not implemented with model.extract_freatures, so this is extremely slow (would take ~256 hours to run on asset_test)

Changes to Fairseq's BART model (model.py and hub_interface.py):
- added `return_encoder_and_decoder_hidden_states` to extract_features
- added `encoder_embedding` to encoder_out

"""

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import fairseq
from fairseq.models.bart import BARTModel

from utils import read_lines, construct_multi_label_dataset

checkpoint_path = '/scratch/tkew/ctrl_tokens/resources/models/muss_en_mined/model.pt'
# infile = '/scratch/tkew/ctrl_tokens/resources/data/examples.en'
infile = '/scratch/tkew/ctrl_tokens/resources/data/en/aligned/asset_test.tsv'
outpath = Path('/scratch/tkew/ctrl_tokens/resources/encoded_sentences/asset_test/')
device = 'cuda'

outpath.mkdir(exist_ok=True, parents=True)

def load_bart_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    hub_interface = torch.hub.load("pytorch/fairseq", "bart.large.cnn").eval()
    hub_interface.model.load_state_dict(sd["model"])
    return hub_interface


bart = load_bart_checkpoint(checkpoint_path)
bart.model.upgrade_state_dict(bart.model.state_dict())
if device:
    bart.to(device)

sentences = read_lines(infile)
sentences, labels = construct_multi_label_dataset(sentences)

# breakpoint()

with open(outpath / 'raw_sents.txt', 'w', encoding='utf8') as raw_file:
    with open(outpath / 'param_labels.txt', 'w', encoding='utf8') as label_file:
        with open(outpath / 'hidden_states.txt', 'w', encoding='utf8') as state_file:

            # breakpoint()
            for sentence, label in tqdm(zip(sentences, labels), total=len(sentences)):
                raw_file.write(f'{sentence}\n')
                label.tofile(label_file, sep='', format='%s')

                tokens = bart.encode(sentence).unsqueeze(0)
                if device:
                    tokens = tokens.to(device)

                # we set return_all_hiddens to get these at intermediary steps 
                # we also set out modified flag to get these encoder hidden states as part of the output
                # by default, return_all_hiddens=True only gives the decoder hiddens
                # TODO: implement batching
                ex_features = bart.extract_features(tokens, return_all_hiddens=True, return_encoder_and_decoder_hidden_states=True)
                ex_features = np.concatenate([ft.detach().cpu().numpy() for ft in ex_features])
                
                # raw_file.write(f'{ex_features.to}\n')
                ex_features.tofile(state_file, sep='', format='%s')