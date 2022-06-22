# Influence of prefix tokens for controlled text generation

## Setup

```
conda create -n env python=3.8
conda activate env
pip install -r requirements.txt

git clone git@github.com:facebookresearch/muss.git
cd muss/
pip install -e .  # Install package
python -m spacy download en_core_web_md
```

## ACCESS

`#TODO`

## Converting pretrained ACCESS models to Hugging Face

To facilitate analysing attention weights, we convert the models trained with Fairseq to Hugging Face. This allows us to easily inspect attention weights with off-the-shelf tools such as `bertviz`.

```
python custom_scripts/convert_bart_original_pytorch_checkpoint_to_pytorch.py \
    resources/models/muss_en_mined/model.pt \
    resources/models/muss_en_mined_hf \
    --hf_config facebook/bart-large-cnn
```

## Probing Experiments

To see if the model retains the information provided by the control tokens, we probe various hidden layer states with a classifier.

If information about the target length is contained in the hidden layer's state, a classifier should be able to correclty identify the appropriate label.

We implement this in Hugging Face, with modifications to the original `src/transformers/models/bart/modeling_bart.py`.

```
python probe.py \
    --infile /scratch/tkew/ctrl_tokens/resources/data/en/aligned/asset_test.tsv \
    --model_path resources/models/muss_en_mined_hf/ \
    --do_train --do_eval --do_predict \
    --early_stopping \
    --wandb \
    --encoder_layers 6 \
    --decoder_layers 0 \
    --aggregate_embeddings 'avg'
```

Alternatively, to submit a job run with slurm:

```
sbatch run_probe.sh -e 6 -d 0 -a avg
```

