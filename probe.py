#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example Usages:

    python probe.py --seed 42 --step_size 0.25 --range_min 0.25 --range_max 1.5 --aggregate_embeddings 'def'

This script implements a simple probe for the BART model.

We use the classifcation head from huggingface to predict the parameter value of a control token.

Makes use of the following changes to the BART model:
    - compute the sentence representation by averaging embeddings (optional)
    - skipping computation for layers that are not considered. This is useful for speeding up processing if we are only interested in an earlier hidden state.
    (https://stackoverflow.com/questions/69835532/dropping-layers-in-transformer-models-pytorch-huggingface)

"""


import os
import random
import logging
import argparse
import pandas as pd
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from utils import read_lines, construct_single_label_dataset, construct_multi_label_dataset

import wandb
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
from datasets import Dataset, DatasetDict, load_metric
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    BartForConditionalGeneration, 
    BartTokenizer,
    BartForSequenceClassification,
    TrainingArguments, 
    Trainer,
    )
from transformers.trainer_callback import EarlyStoppingCallback

logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', required=False, default='resources/data/examples.en', help='')
    ap.add_argument('--model_path', required=False, default='facebook/bart-base', help='')
    ap.add_argument('--output_dir', required=False, default='resources/models/classifiers', help='')
    ap.add_argument('--device', required=False, default='cuda', help='')
    
    ap.add_argument('--ctrl_token', required=False, 
        help='If provided, constructs a dataset with incremental \
            parameter values only for the specified control token. \
            If not provided, all control tokens are tested in a multi-label classification task.')
    ap.add_argument('--seed', required=False, type=int, default=42, help='')
    ap.add_argument('--step_size', required=False, type=float, default=0.25, 
        help='step size between parameter values'
        )
    ap.add_argument('--range_min', required=False, type=float, default=0.25, 
        help='minimum parameter value used'
        )
    ap.add_argument('--range_max', required=False, type=float, default=1.5,  
        help='maximum parameter value used'
        )
    ap.add_argument('--aggregate_embeddings', required=False, type=str, default='def', choices=['def', 'avg'], 
        help='whether or not to compute the sentence representation by averaging embeddings. \
            If not specified, default computation of sentence representation is used.'
        )

    # arguments passed directly to the trainer
    ap.add_argument('--batch_size', required=False, type=int, default=256, help='')
    ap.add_argument('--num_workers', required=False, type=int, default=4, help='')
    ap.add_argument('--early_stopping', required=False, action='store_true', help='')
    ap.add_argument('--early_stopping_patience', required=False, type=int, default=5, help='')
    ap.add_argument('--early_stopping_threshold', required=False, type=float, default=0.0001, help='')
    ap.add_argument('--do_train', required=False, action='store_true', help='')
    ap.add_argument('--do_eval', required=False, action='store_true', help='')
    ap.add_argument('--do_predict', required=False, action='store_true', help='')
    ap.add_argument('--wandb', required=False, action='store_true', help='')
    ap.add_argument('--debug', required=False, action='store_true', help='')
    ap.add_argument('--num_train_epochs', required=False, type=int, default=20, help='')
    ap.add_argument('--lr', required=False, type=float, default=2e-05, help='')

    ap.add_argument('--encoder_layers', required=False, type=int, default=-1,
        help='Number of consecutive encoder layers used to compute the hidden \
            representation that is given to the decoder. \
            Default = -1 indicates ALL encoder layers are active.'
            )
    ap.add_argument('--decoder_layers', required=False, type=int, default=-1, 
        help='Number of consecutive decoder layers used to compute the hidden \
            representation that is given to the classification head. \
            Default = -1 indicates ALL decoder layers are active.'
        )

    return ap.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def construct_dataset(inputs, labels, seed, use_one_hot=False):

    if use_one_hot:
        enc = OneHotEncoder(handle_unknown='error')
        labels = enc.fit_transform([l for l in labels]).toarray()

    data = Dataset.from_dict({'text': inputs, 'labels': labels})
    # 80% train, 10% test + validation
    train_testvalid = data.train_test_split(shuffle=True, seed=seed, test_size=0.1)
    # Split the 20% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(shuffle=False, seed=seed, test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train'],
        })
    
    return train_test_valid_dataset


def train_classifier_layer(model, tokenizer, dataset, args):

    acc = load_metric("accuracy")
    
    def compute_metrics(eval_pred):
    
        acc_scores = {}

        predictions, labels = eval_pred
        logits = predictions[0]

        if args.ctrl_token is None: # assumes all 4 distinct control tokens are being evaluated
            # the order is defined in utils.py as 'len_ratio', 'lev_sim', 'word_rank', 'tree_depth'
            # TODO remove hardcoded value of 5 - should be ~ labels.shape[1] // 4
            logits = np.expand_dims(logits, axis=1).reshape(-1, 4, 5)
            labels = np.expand_dims(labels, axis=1).reshape(-1, 4, 5)
            prediction_probs = softmax(logits, axis=-1)            
            for i, param in enumerate(['len_ratio', 'lev_sim', 'word_rank', 'tree_depth']):
                acc_scores[f'acc:{param}'] = acc.compute(
                    predictions=prediction_probs[:, i, :].argmax(axis=-1),
                    references=labels[:, i, :].argmax(axis=-1)
                    )['accuracy']
                acc_scores['acc:all'] = acc.compute(predictions=prediction_probs.argmax(axis=-1).flatten(), references=labels.argmax(axis=-1).flatten())['accuracy']

        else: # a specific control token is being evaluated in isolation
            prediction_probs = softmax(logits, axis=-1)
            acc_scores[f'acc:{args.ctrl_token}'] = acc.compute(predictions=prediction_probs.argmax(axis=-1).flatten(), references=labels.argmax(axis=-1).flatten())['accuracy']
        
        return acc_scores

    model_name = Path(model.config._name_or_path).stem
    
    active_enc_layers_str = ''.join(['1' if i < model.config.active_enc_layers and model.config.active_enc_layers != 0 else '0' for i in range(model.config.encoder_layers)])
    active_dec_layers_str = ''.join(['1' if i < model.config.active_dec_layers and model.config.active_dec_layers != 0 else '0' for i in range(model.config.decoder_layers)])    
    
    sentence_representation = 'avg' if model.config.average_embeddings else 'def'

    tgt_ctrl_token = 'all' if not args.ctrl_token else args.ctrl_token

    model_name = f'{model_name}-{tgt_ctrl_token}-{sentence_representation}-{active_enc_layers_str}-{active_dec_layers_str}'

    print(f'*** Running model: {model_name} ***')

    output_dir = Path(args.output_dir) / model_name

    training_callbacks = None
    if args.early_stopping:
        training_callbacks = [EarlyStoppingCallback(args.early_stopping_patience, args.early_stopping_threshold)]

    if args.wandb and not args.debug:
        wandb.init(
            project="ctrl_tokens_probe", 
            name=model_name,
            tags=[tgt_ctrl_token, sentence_representation, active_enc_layers_str, active_dec_layers_str],
            group=sentence_representation
            )

    training_args = TrainingArguments(
        output_dir,
        overwrite_output_dir=True,
        run_name=model_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=0.0, 
        warmup_steps=0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_accumulation_steps=6,
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        push_to_hub=False,
        report_to="wandb" if args.wandb and not args.debug else "none",
        num_train_epochs=args.num_train_epochs,
        save_total_limit=1,
        no_cuda=True if args.device == 'cpu' else False,
        seed=args.seed, data_seed=args.seed,
        # debug=args.debug,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=training_callbacks,
    )   

    if args.do_train:
        # breakpoint()
        trainer.train()
    
    if args.do_eval:
        trainer.evaluate()

    if args.do_predict:
        trainer.predict(test_dataset=dataset["test"])

    return 
    

def main(args):

    seed_everything(args.seed)

    if args.debug:
        os.environ["WANDB_DISABLED"] = "true"

    # if args.wandb and not args.debug:
    #     wandb.init(project="ctrl_tokens_probe")

    logger = logging.getLogger(__name__)

    device = args.device #'cuda' if torch.cuda.is_available() else 'cpu'
    infile = args.infile # 'resources/data/examples.en'
    model_path = args.model_path # "facebook/bart-base"

    # build dataset
    sentences = read_lines(infile)

    if args.ctrl_token:
        inputs, labels = construct_single_label_dataset(sentences, args.ctrl_token, args.range_min, args.range_max, args.step_size)
        dataset = construct_dataset(inputs, labels, seed=args.seed, use_one_hot=True)
    else:
        inputs, labels = construct_multi_label_dataset(sentences, args.range_min, args.range_max, args.step_size)
        dataset = construct_dataset(inputs, labels, seed=args.seed, use_one_hot=False)
    

    #### prepare model for experiments
    if len(dataset['train']['labels'][0]) > 1:
        num_labels = len(dataset['train']['labels'][0]) # if labels are 1-hot encoded
    else:
        num_labels = len(set([i[0] for i in dataset['train']['labels']])) # if labels are categorical

    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, output_hidden_states=True)

    logger.info(f'Loaded {model_path} on {model.device}...')

    for name, param in model.named_parameters():
        if not name.startswith("classification"): # choose whatever you like here
            logger.info(f'Freezing parameter: {name}')
            param.requires_grad = False
        else:
            logger.info(f'*** {name} parameter weights will be trained! ***')

    # ensure that dropout is not used while training the classifier head
    model.config.dropout = 0.0
    model.config.attention_dropout = 0.0

    if args.aggregate_embeddings == 'avg':
        model.config.average_embeddings = True
        logger.info('*** Note, hidden states will be averaged to compute the sentence representation ***')
    else:
        model.config.average_embeddings = False

    if args.encoder_layers == -1: # all available layers are active, e.g. 12 for bart-large
        model.config.active_enc_layers = model.config.encoder_layers
    else:
        model.config.active_enc_layers = args.encoder_layers

    if args.decoder_layers == -1: # all available layers are active, e.g. 12 for bart-large
        model.config.active_dec_layers = model.config.decoder_layers
    else:
        model.config.active_dec_layers = args.decoder_layers

    # check validitiy
    if model.config.active_dec_layers > 0 and model.config.active_enc_layers < 12:
        raise RuntimeError('Layers are not contiguous! All encoder layers must be active if decoder layers are active.')
    
    # https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb    
    def preprocess_function(examples):
        return tokenizer(examples['text'], max_length=512, truncation=True)
    
    dataset = dataset.map(preprocess_function, batched=True) #, num_proc=args.num_workers)
    # breakpoint()
    train_classifier_layer(model, tokenizer, dataset, args)

if __name__ == "__main__":
    args = parse_args()
    main(args)