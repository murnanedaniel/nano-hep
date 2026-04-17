#!/usr/bin/env bash
# nano-hep pilot: 100k events, 5k steps, 1 GPU, ~1h.
# Usage: bash scripts/pilot_1gpu.sh [JOBID]
set -uo pipefail
REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep"
PYTHON="/global/homes/d/danieltm/.conda/envs/influencer/bin/python"
cd "${REPO}"
unset TMPDIR
if [ -n "${1:-}" ]; then
  JOBID="$1"
  srun --jobid=${JOBID} --exact -N1 -n1 --gpus-per-task=1 \
    bash -c "cd ${REPO} && unset TMPDIR && ${PYTHON} -u -m nano_hep.train_hep --config configs/nano_hep_pilot.yml" \
    2>&1 | tee runs/pilot.log
else
  ${PYTHON} -u -m nano_hep.train_hep --config configs/nano_hep_pilot.yml 2>&1 | tee runs/pilot.log
fi
