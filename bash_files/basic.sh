#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-bigbayes-test
#SBATCH --nodelist=zizgpu06.cpu.stats.ox.ac.uk
#SBATCH --gres=gpu:10
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00:00:00
#SBATCH --mem=700G
#SBATCH --ntasks=1

# For job arrays we need to use "slurm-%A-%a" pattern.
#SBATCH --output=/vols/bitbucket/caalfano/logs/slurm-%A_%a.out
#SBATCH --error=/vols/bitbucket/caalfano/logs/slurm-%A_%a.out

export MPLCONFIGDIR="/data/ziz/not-backed-up/caalfano/.config/matplotlib"
export WANDB_DIR=/vols/bitbucket/caalfano/wandb
/data/ziz/not-backed-up/caalfano/miniconda3/bin/activate llm-env
cd /data/ziz/caalfano/direct-preference-optimization
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16