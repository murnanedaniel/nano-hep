#!/usr/bin/env bash
set -uo pipefail
REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep"
PYTHON="/global/homes/d/danieltm/.conda/envs/influencer/bin/python"
cd "${REPO}"
unset TMPDIR
JOBID=${1:?Usage: $0 JOBID}
srun --jobid=${JOBID} --exact -N1 -n1 --gpus-per-task=1 \
  bash -c "cd ${REPO} && unset TMPDIR && WANDB__SERVICE_WAIT=120 ${PYTHON} -u -m nano_hep.train_hep --config configs/nano_hep_smoke_q3.yml" \
  2>&1 | tee runs/smoke_q3.log
