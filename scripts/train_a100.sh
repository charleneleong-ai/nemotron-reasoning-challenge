#!/usr/bin/env bash
#
# train_a100.sh — fully-detached A100 LoRA training launcher.
#
# RUN THIS ON THE A100 BOX, NOT ON A MAC. There is no GPU here on the dev Mac.
#
# Prerequisites on the A100 host:
#   - repo cloned + `uv` installed (https://docs.astral.sh/uv/)
#   - .env contains KAGGLE_API_TOKEN (for `main download` / `main submit`)
#   - Hugging Face auth for the gated base model
#     (`huggingface-cli login`, or HF_TOKEN exported)
#   - CUDA drivers + nvidia-smi available (custom-code Mamba-MoE needs CUDA)
#
# The pipeline + watchdog are launched as detached daemons (setsid + nohup + disown)
# so they survive SSH disconnect AND this shell dying. Verify after launch that
# `ps -ef | grep train_a100` shows the pipeline with PPID=1 (init-owned).
#
# Usage:
#   ./scripts/train_a100.sh            # default train preset = lora_a100
#   ./scripts/train_a100.sh qlora_t4   # override the train preset

set -euo pipefail

TRAIN_PRESET="${1:-lora_a100}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p logs

TRAIN_LOG="logs/train_a100_${TS}.log"
WATCHDOG_LOG="logs/gpu_watchdog_${TS}.jsonl"
WATCHDOG_OUT="logs/gpu_watchdog_${TS}.out"

# eval + package read the adapter from cfg.train.output_dir, so they MUST carry the
# same train=${TRAIN_PRESET} override as training — otherwise they look in the
# default (smoke) output dir and miss the trained adapter.
PIPELINE="uv sync --extra gpu \
  && uv run main download \
  && uv run main train model=nemotron_nano train=${TRAIN_PRESET} \
  && uv run main eval model=nemotron_nano train=${TRAIN_PRESET} \
  && uv run main package train=${TRAIN_PRESET}"

echo "Launching training pipeline (preset=${TRAIN_PRESET})..."
setsid nohup bash -c "${PIPELINE}" </dev/null >>"${TRAIN_LOG}" 2>&1 &
disown
TRAIN_PID=$!

echo "Launching GPU watchdog for PID ${TRAIN_PID}..."
setsid nohup python -u scripts/gpu_watchdog.py \
  --pid "${TRAIN_PID}" --log "${WATCHDOG_LOG}" --gpu-index 0 \
  </dev/null >>"${WATCHDOG_OUT}" 2>&1 &
disown
WATCHDOG_PID=$!

cat <<EOF

Launched detached daemons:
  pipeline PID : ${TRAIN_PID}   log: ${TRAIN_LOG}
  watchdog PID : ${WATCHDOG_PID}   log: ${WATCHDOG_LOG} (kill reasons), ${WATCHDOG_OUT} (stdout)

Verify both survived this shell (expect PPID = 1, init-owned):
  ps -ef | grep -E 'train_a100|gpu_watchdog' | grep -v grep

Tail progress:
  tail -f ${TRAIN_LOG}
EOF
