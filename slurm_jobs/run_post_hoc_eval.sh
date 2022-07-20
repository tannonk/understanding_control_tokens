#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --output=/data/tkew/projects/ctrl_tokens/logs/%j.out

# Author: T. Kew (inspired by N. Spring)
# sbatch slurm_jobs/run_probe.sh -e -1 -d 0

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

base="/data/tkew/projects/ctrl_tokens"
dataset="$base/resources/data/en/aligned/asset_test.tsv"

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


for config in "111000000000-000000000000" "111111000000-000000000000" "111111111000-000000000000" "111111111111-000000000000" "111111111111-111000000000" "111111111111-111111000000" "111111111111-111111111000" "111111111111-111111111111"
do
    model_path="$base/resources/models/classifiers/muss_en_mined_hf-all-def-$config"
    # -s FILE exists and has a size greater than zero
    if [ -s "$model_path/test_results.json" ]; then
        echo "$model_path already evaluated"
    else
        python probe.py \
            --infile $dataset \
            --model_path $model_path \
            --do_eval --do_predict
    fi
done

# python probe.py \
#     --infile resources/data/en/aligned/asset_test.tsv \
#     --model_path resources/models/classifiers/muss_en_mined_hf-all-def-111000000000-000000000000 \
#     --do_eval --do_predict

# python probe.py \
#     --infile resources/data/en/aligned/asset_test.tsv \
#     --model_path resources/models/classifiers/muss_en_mined_hf-all-def-111111000000-000000000000 \
#     --do_eval --do_predict

# python probe.py \
#     --infile resources/data/en/aligned/asset_test.tsv \
#     --model_path resources/models/classifiers/muss_en_mined_hf-all-def-111111111000-000000000000 \
#     --do_eval --do_predict

# python probe.py \
#     --infile resources/data/en/aligned/asset_test.tsv \
#     --model_path resources/models/classifiers/muss_en_mined_hf-all-def-111111111111-000000000000 \
#     --do_eval --do_predict

# python probe.py \
#     --infile resources/data/en/aligned/asset_test.tsv \
#     --model_path resources/models/classifiers/muss_en_mined_hf-all-def-111111111111-111000000000 \
#     --do_eval --do_predict

# python probe.py \
#     --infile resources/data/en/aligned/asset_test.tsv \
#     --model_path resources/models/classifiers/muss_en_mined_hf-all-def-111111111111-111111000000 \
#     --do_eval --do_predict

# python probe.py \
#     --infile resources/data/en/aligned/asset_test.tsv \
#     --model_path resources/models/classifiers/muss_en_mined_hf-all-def-111111111111-111111111000 \
#     --do_eval --do_predict

# python probe.py \
#     --infile resources/data/en/aligned/asset_test.tsv \
#     --model_path resources/models/classifiers/muss_en_mined_hf-all-def-111111111111-111111111111 \
#     --do_eval --do_predict