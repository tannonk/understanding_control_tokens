#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:Tesla-V100-32GB:1
#SBATCH --partition=volta
#SBATCH --output=/data/tkew/projects/ctrl_tokens/logs/%j.out

# Author: T. Kew (inspired by N. Spring)
# sbatch jobs/run_probe.sh -e -1 -d 0

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

base="/data/tkew/projects/ctrl_tokens"
infile="$base/resources/data/en/aligned/asset_test.tsv" # default
output_dir="$base/resources/models/classifiers/asset_test"
model_path="$base/resources/models/muss_en_mined_hf/"
agg_method="def"
encoder_layers=""
decoder_layers=""

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -b [base] -e [encoder_layers] -d [decoder_layers] -a [agg_method] -i [infile] -m [model_path]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "b:e:d:a:i:m" flag; do
  case "${flag}" in
    b) base="$OPTARG" ;;
    e) encoder_layers="$OPTARG" ;;
    d) decoder_layers="$OPTARG" ;;
    a) agg_method="$OPTARG" ;;
    i) infile="$OPTARG" ;;
    m) model_path="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $encoder_layers ]]; then
    print_missing_arg "[-e encoder_layers]" "number of active encoder layers must be specified"
    exit 1
fi
if [[ -z $decoder_layers ]]; then
    print_missing_arg "[-e decoder_layers]" "number of active decoder layers must be specified"
    exit 1
fi

#######################################################################
# ACTIVATE ENV
#######################################################################

module purge
module load volta anaconda3
module list

eval "$(conda shell.bash hook)"
conda deactivate
conda activate ctrl_tokens && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

python $base/probe.py \
    --infile $infile \
    --output_dir $output_dir \
    --model_path $model_path \
    --do_train \
    --do_eval \
    --do_predict \
    --early_stopping \
    --wandb \
    --encoder_layers $encoder_layers \
    --decoder_layers $decoder_layers \
    --aggregate_embeddings $agg_method
