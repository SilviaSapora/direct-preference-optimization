#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-bigbayes-test
#SBATCH --nodelist=zizgpu06.cpu.stats.ox.ac.uk
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=02-00:00:00
#SBATCH --mem=700G
#SBATCH --ntasks=1
# SBATCH --mail-user=silvia.sapora@stats.ox.ac.uk     # set email address to use, change to your own email address instead of "me"
#SBATCH --mail-type=ALL                   # Caution: fine for debug, but not if handling hundreds of jobs!

# For job arrays we need to use "slurm-%A-%a" pattern.
#SBATCH --output=/vols/bitbucket/sapora/logs/slurm-%A_%a.out
#SBATCH --error=/vols/bitbucket/sapora/logs/slurm-%A_%a.out


# export MPLCONFIGDIR="/data/ziz/not-backed-up/sapora/.config/matplotlib"
export WANDB_DIR=/vols/bitbucket/sapora/wandb
/data/ziz/not-backed-up/sapora/miniconda3/bin/activate llm-env
cd /data/ziz/sapora/direct-preference-optimization
# python -u train.py exp_name=anthropic_sft_pythia69 model=pythia28 datasets=[hh] loss=sft gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16

# python -u train.py exp_name=anthropic_dpo_pythia69 model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/vols/bitbucket/sapora/.cache/sapora/anthropic_sft_pythia69_2024-03-17_16-27-41_731972/step-159744/policy.pt

python -u train.py exp_name=anthropic_bdpo_pythia69 model=pythia28 datasets=[hh] loss=b-dpo loss.beta=0.1 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/vols/bitbucket/sapora/.cache/sapora/anthropic_sft_pythia69_2024-03-17_16-27-41_731972/step-159744/policy.pt
