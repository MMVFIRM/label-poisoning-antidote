#!/usr/bin/env bash
set -euo pipefail

# One-seed, paper-faithful sanity run against the public FairMean repository.
# This intentionally avoids the 30-run matrix used by the optional full script.

REPO="${REPO:-FairnessUnderLP}"
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/Zhg9300/FairnessUnderLP.git "$REPO"
fi
cd "$REPO"
python -m pip install -r requirements.txt

DEVICE="${DEVICE:-0}"
ROUNDS="${ROUNDS:-3000}"
SEED="${SEED:-1}"
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
  --seed "$SEED"
  --partition_seed "$SEED"
)

python run.py "${COMMON[@]}" --algorithm FedAvg \
  --attack_mode label_targeted_flip --dishonest_num 2

python run.py "${COMMON[@]}" --algorithm FairMean \
  --fairmean_kappa 1 --fairmean_tau 1 \
  --attack_mode label_targeted_flip --dishonest_num 2

python run.py "${COMMON[@]}" --algorithm qFedAvg \
  --q 1 --qffl_update_rule objective_gradient \
  --attack_mode label_targeted_flip --dishonest_num 2

printf '\nOne-seed external sanity run complete. Inspect the FairnessUnderLP results directory.\n'
