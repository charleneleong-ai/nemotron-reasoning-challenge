#!/usr/bin/env bash
# Wait for the in-flight augmented SFT to exit, then run the leak-free per-category
# val eval on sft_valexcl (the val-excluded adapter). Detached daemon; logs timestamped.
set -u
TRAIN_PID="${1:?train pid}"
MODEL=/home/ubuntu/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
cd /home/ubuntu/nemotron-reasoning-challenge

while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
sleep 30  # let the trainer flush + free VRAM

TS=$(date -u +%Y%m%dT%H%M%SZ)
.venv-unsloth/bin/python -u scripts/eval_val.py \
  --adapter adapters/sft_valexcl \
  --model-path "$MODEL" \
  --data data/train.csv \
  >> "logs/eval_valexcl_${TS}.log" 2>&1
