# Influence of prefix tokens for controlled text generation

This project aims to better understand the influence of control tokens when applied to sentence simplification. Specifically, we analyse the MUSS model proposed by [Martin et al. (2021)](https://arxiv.org/abs/2005.00352) which is trained on English paraphrase data mined from the web.

This model relies on the ACCESS control method which was proposed in [Martin et al. (2020)](https://aclanthology.org/2020.lrec-1.577/).

Both the ACCESS and MUSS models use four consecutive control tokens prepended to the source text in order to control aspects of the generation. 

This rather simple approach has a couple of shortcomings:
    1. it is not clear to what extent the control tokens actually steer the generation process (e.g. do all control tokens contribute to a simplification or is one just doing the heavy lifting or are there interaction effects that cancel out control tokens, or worse, work against each other)
    2. given an input sentence to simplify, how do we choose the most suitable control token values for good results

This readme provides some descriptive information for running our experiments.

This Google Slides [https://docs.google.com/presentation/d/1gBmOx5T8p2UkWpOefZFMjsyJhl5OZ3N4ERV5A_pX4OE/edit?usp=sharing](presentation) contains a brief overview of background descriptions and preliminary results.


## Setup

NOTE: To run ALTI+, you need Fairseq v0.12.1, while MUSS models uses Fairseq v0.10.2. The easiest way to set this up and avoid dependency issues is t run two separate environments, e.g.

```
# for main experiments
conda create -n ctrl_tokens python=3.8
conda activate ctrl_tokens
pip install -r envs/ctrl_tokens_requirements.txt

# for attention analyses
conda create -n alti_plus python=3.8
conda activate alti_plus
pip install -r envs/alti_plus_requirements.txt
```

<!-- Alternatively, could also try a single env (at own risk). Upgrading Fairseq doesn't seem to cause any issues when running inference with MUSS, however, this has not been thoroughly tested.

```
conda create -n ctrl_tokens_v2 python=3.8
conda activate ctrl_tokens_v2
pip install -r envs/alti_plus_requirements.txt
pip install -r envs/ctrl_tokens_requirements.txt

``` -->

## Converting pretrained ACCESS/MUSS models to Hugging Face

To facilitate some of our analyses, we convert the models trained with Fairseq to Hugging Face. This allows us to easily build probing models and inspect attention weights with off-the-shelf tools such as `bertviz`.

```
python muss/custom_scripts/convert_bart_original_pytorch_checkpoint_to_pytorch.py \
    resources/models/muss_en_mined/model.pt \
    resources/models/muss_en_mined_hf \
    --hf_config facebook/bart-large-cnn
```

## Surface-level Analysis

The most obvious way to inspect the effect of the control tokens is to generate simplifications with a range of possible values and look at the generated outputs.

We do this with both in-distribution and out-of-distribution control token values. If the control tokens do their job, we would expect to see the attributes of the generated outputs to reflect (or correlate strongly) with the specified values given as input to the model.

To run these experiments, we provide the notebook [./notebooks/surface_level_analysis.ipynb](surface_level_analysis.ipynb) in which we construct a large inference dataset with input sentences hardcoded with a range of possible control token values and then generate simplifications using our Hugging Face ported model.

The correlation heatmaps in [./results/plots/] show that the correlation for in-domain parameter values and output attributes is low (~0.3) for three out of the four control tokens. This is similar to the preliminary results gained by Iza Škrjanec.

## Model Analysis

### Probing Experiments

To see if the model retains the information provided by the control tokens, we probe various hidden layer states with a classifier.

If information about the target length is contained in the hidden layer's state, a classifier should be able to correclty identify the appropriate label.

We implement this in Hugging Face, with modifications to the original `src/transformers/models/bart/modeling_bart.py`.

```
python probe.py \
    --infile /scratch/tkew/ctrl_tokens/resources/data/examples.en \
    --model_path resources/models/muss_en_mined_hf/ \
    --do_train --do_eval --do_predict \
    --early_stopping \
    --encoder_layers 6 \
    --decoder_layers 0 \
    --aggregate_embeddings 'avg'
```

Alternatively, to submit a job run with slurm:

```
sbatch slurm_jobs/run_probe.sh -e 6 -d 0 -a avg
```

Evaluating a probe on the test set should happen automatically once training is complete. However, to do this manually (i.e. post hoc), specify the existing model's directory as `--model_path`, e.g.:

```
python probe.py \
    --infile /scratch/tkew/ctrl_tokens/resources/data/examples.en \
    --model_path resources/models/classifiers/muss_en_mined_hf-all-avg-111111000000-000000000000 \
    --do_eval --do_predict
```

### Control Token Contirbutions

**ALTI+** [Ferrando et al. (2022)](https://arxiv.org/abs/2203.04212) was proposed to measure the *contributions* of source and prefix tokens on subsequent generation timesteps.

We consider this method to analyse the contribution of the MUSS model's control tokens.

The notebook [./transformer-contributions-nmt/muss_interprebility.ipynb] can be used to analyse attention weights in the custom BART model trained with Fairseq.


<!-- 

## handy commands

```
python simplify_file.py \
    /scratch/tkew/ctrl_tokens/resources/data/examples.en \
    --out_path /scratch/tkew/ctrl_tokens/resources/data/examples.en.decoded \
    --model_name muss_en_mined
``` 

-->