#!/usr/bin/env bash
# Vendor the competition csv into the env package (gitignored) and push to Prime.
#
#   scripts/prime_env_push.sh        # -> chaleong/nemotron-reasoning
#
# The env reads data/train.csv from its own dir so the pushed wheel is self-contained;
# we copy it in here rather than commit a 3MB duplicate of data/train.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=environments/nemotron_reasoning
mkdir -p "$ENV/data"
cp data/train.csv "$ENV/data/train.csv"
echo "vendored $(wc -l < "$ENV/data/train.csv") rows into $ENV/data/"
exec prime env push --path "$ENV"
