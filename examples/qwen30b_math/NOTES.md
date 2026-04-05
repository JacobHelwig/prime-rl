# Quick Start


## Create env 

```bash
uv sync --all-extras
```

## Add to .env

`jacob.a.helwig_tamu.edu@slogin-01:/scratch/user/jacob.a.helwig_tamu.edu/prime-rl$ cat .env `

```bash
WANDB_API_KEY=...
HUGGINGFACE_TOKEN=...
HF_HOME=/scratch/user/jacob.a.helwig_tamu.edu/primerl_data
TRITON_CACHE_DIR=/scratch/user/jacob.a.helwig_tamu.edu/primerl_data/triton_cache
```

# Train 

## Start tmux

```bash
bash scripts/slurm_tmux.sh hendrycks-math-qwen0_8b /scratch/user/jacob.a.helwig_tamu.edu/prime-rl/logs/tmux
```

## Start train

```bash
uv run rl @ configs/hendrycks_math/rl.toml --output-dir /scratch/user/jacob.a.helwig_tamu.edu/prime-rl/logs/slurm
```


