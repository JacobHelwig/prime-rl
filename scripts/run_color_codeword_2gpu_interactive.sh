#!/usr/bin/env bash

set -euo pipefail

JOB_ID="${1:-${JOB_ID:-}}"
if [[ -z "$JOB_ID" ]]; then
  echo "Usage: $0 <job_id>" >&2
  echo "   or: JOB_ID=<job_id> $0" >&2
  exit 1
fi

REPO_DIR=/scratch/user/jacob.a.helwig_tamu.edu/prime-rl
DATA_PATH=/scratch/user/jacob.a.helwig_tamu.edu/primeRL_data
CONFIG="${CONFIG:-configs/multimodal/rl_color_codeword.toml}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/user/jacob.a.helwig_tamu.edu/prime-rl/logs/rl_color_codeword_2gpu_main}"

exec srun \
  --jobid="$JOB_ID" \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:2 \
  bash -lc "
    export HF_HOME=$DATA_PATH
    export TRITON_CACHE_DIR=$DATA_PATH/triton_cache
    mkdir -p \"\$HF_HOME\" \"\$TRITON_CACHE_DIR\"
    cd $REPO_DIR
    uv run rl @ $CONFIG \
      --max_steps 3 \
      --orchestrator.batch_size 16 \
      --orchestrator.rollouts_per_example 2 \
      --orchestrator.train.sampling.max-completion-tokens 32 \
      --output-dir $OUTPUT_DIR
  "
