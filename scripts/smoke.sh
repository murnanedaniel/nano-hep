#!/usr/bin/env bash
# nano-hep smoke: 10k events, 500 steps, 1 GPU, ~10 min.
# Usage: bash scripts/smoke.sh [JOBID]  (if JOBID given, runs via srun on that alloc)
set -uo pipefail

REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep"
PYTHON="/global/homes/d/danieltm/.conda/envs/influencer/bin/python"
cd "${REPO}"
unset TMPDIR

if [ -n "${1:-}" ]; then
  JOBID="$1"
  echo "Running via srun on jobid ${JOBID}"
  srun --jobid=${JOBID} --exact -N1 -n1 --gpus-per-task=1 \
    bash -c "cd ${REPO} && unset TMPDIR && ${PYTHON} -u -m nano_hep.train_hep --config configs/nano_hep_smoke.yml" \
    2>&1 | tee runs/smoke.log
else
  echo "Running locally (no srun); assumes a GPU is visible"
  ${PYTHON} -u -m nano_hep.train_hep --config configs/nano_hep_smoke.yml 2>&1 | tee runs/smoke.log
fi
