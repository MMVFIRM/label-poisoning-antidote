#!/usr/bin/env bash
set -euo pipefail

# Gate 33 official FairMean/FedAvg/q-FFL reproduction harness.
#
# Requirements:
#   git
#   Python environment with PyTorch/torchvision compatible with your GPU
#
# This uses the public authors' repository:
#   https://github.com/Zhg9300/FairnessUnderLP
#
# FairMean paper-aligned constraints enforced by the repo:
#   C=1, E=1, sgd_step=False, mean gradient aggregation.
#
# CIFAR-10:
#   N=10 clients
#   Dirichlet alpha=.1
#   pairwise/targeted flip y -> 9-y
#   2 poisoned clients
#
# Five seeds are run for clean and attacked conditions.

REPO="${REPO:-FairnessUnderLP}"
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/Zhg9300/FairnessUnderLP.git "$REPO"
fi
cd "$REPO"

python -m pip install -r requirements.txt

DEVICE="${DEVICE:-0}"
ROUNDS="${ROUNDS:-3000}"
COMMON=(
  --device "$DEVICE"
  --module CNN
  --dataloader DataLoader_cifar10_dir
  --Diralpha 0.1
  --N 10
  --C 1
  --B full
  --R "$ROUNDS"
  --E 1
  --sgd_step False
  --weight_decay 0
  --test_interval 50
)

for SEED in 1 2 3 4 5; do
  PARTITION_SEED="$SEED"

  # Clean FedAvg
  python run.py "${COMMON[@]}" \
    --seed "$SEED" --partition_seed "$PARTITION_SEED" \
    --algorithm FedAvg

  # Attacked FedAvg
  python run.py "${COMMON[@]}" \
    --seed "$SEED" --partition_seed "$PARTITION_SEED" \
    --algorithm FedAvg \
    --attack_mode label_targeted_flip --dishonest_num 2

  # Clean FairMean, repository defaults kappa=tau=1
  python run.py "${COMMON[@]}" \
    --seed "$SEED" --partition_seed "$PARTITION_SEED" \
    --algorithm FairMean \
    --fairmean_kappa 1 --fairmean_tau 1

  # Attacked FairMean
  python run.py "${COMMON[@]}" \
    --seed "$SEED" --partition_seed "$PARTITION_SEED" \
    --algorithm FairMean \
    --fairmean_kappa 1 --fairmean_tau 1 \
    --attack_mode label_targeted_flip --dishonest_num 2

  # Clean q-FFL, q=1, direct objective-gradient mode
  python run.py "${COMMON[@]}" \
    --seed "$SEED" --partition_seed "$PARTITION_SEED" \
    --algorithm qFedAvg \
    --q 1 --qffl_update_rule objective_gradient

  # Attacked q-FFL
  python run.py "${COMMON[@]}" \
    --seed "$SEED" --partition_seed "$PARTITION_SEED" \
    --algorithm qFedAvg \
    --q 1 --qffl_update_rule objective_gradient \
    --attack_mode label_targeted_flip --dishonest_num 2

done

echo "Official baseline runs complete. Inspect results/ and run logs."
