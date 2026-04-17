#!/usr/bin/env bash
# Launch nano-hep 1M training on 4 nodes × 4 GPUs = 16 GPUs via torchrun-under-srun.
# Usage: scripts/overnight_1M_ddp.sh <JOBID>
set -uo pipefail
REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep"
PYTHON="/pscratch/sd/d/danieltm/envs/hep4m2/bin/python"
CFG="${REPO}/configs/nano_hep_overnight_1M_ddp.yml"
cd "${REPO}"
unset TMPDIR
JOBID=${1:?Usage: $0 JOBID}

# Pick a rendezvous host (first node in the allocation)
MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST 2>/dev/null | head -n1)
if [ -z "${MASTER_NODE}" ]; then
  MASTER_NODE=$(squeue -h -j $JOBID -o "%N" | sed 's/nid\[\([0-9]\+\).*/nid\1/' | head -1)
fi
export MASTER_ADDR=${MASTER_NODE}
export MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# 4 nodes × 4 gpus/node.
# srun spawns 1 torchrun per node, torchrun spawns 4 local workers per node.
srun --jobid=${JOBID} --nodes=4 --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-node=4 \
  bash -c "cd ${REPO} && unset TMPDIR && \
    WANDB__SERVICE_WAIT=300 NCCL_SOCKET_IFNAME=\"^lo,docker0\" \
    ${PYTHON} -m torch.distributed.run \
      --nproc_per_node=4 --nnodes=4 \
      --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      --node_rank=\$SLURM_NODEID \
      -m nano_hep.train_hep --config ${CFG}" \
  2>&1 | tee runs/overnight_1M_ddp.log
