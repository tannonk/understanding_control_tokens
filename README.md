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

##

