#!/bin/bash
# Chained interactive allocations for nano-hep on the full 89M dataset.
# Same pattern as HEP4M's chain_interactive.sh: 4h interactive alloc → train
# under torchrun → on SLURM timeout, reallocate and resume from last.ckpt.
#
# Usage:
#   bash scripts/chain_interactive.sh [OUT_DIR]
#
# If OUT_DIR is omitted, uses cfg.training.out_dir (runs/nano_hep_89M_ddp).
# OUT_DIR is used as --resume_from target on every iteration after the first.
#
# Runs under tmux/nohup to survive disconnect. Exits when training returns 0
# (max_steps reached).

set -uo pipefail

REPO=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep
PYTHON=/pscratch/sd/d/danieltm/envs/hep4m2/bin/python
CFG=$REPO/configs/nano_hep_89M_ddp.yml
OUT_DIR="${1:-$REPO/runs/nano_hep_89M_ddp}"
LOGDIR=$REPO/runs/chain_logs
JOURNAL=$LOGDIR/chain_journal.log
mkdir -p "$LOGDIR" "$OUT_DIR"

cd "$REPO"
unset TMPDIR

iter=0
while true; do
  iter=$((iter+1))
  echo "[chain] === iter $iter @ $(date) ===" | tee -a "$JOURNAL"

  # --- allocate 4h interactive -----------------------------------------------
  SALLOC_OUT=$(mktemp)
  salloc --no-shell \
    -N 4 -n 4 --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-node=4 \
    -C "gpu&hbm80g" -A m3443_g -t 4:00:00 -q interactive \
    > "$SALLOC_OUT" 2>&1
  cat "$SALLOC_OUT" | tee -a "$JOURNAL"
  JOBID=$(grep -oE "job allocation [0-9]+" "$SALLOC_OUT" | tail -1 | awk '{print $3}')
  rm -f "$SALLOC_OUT"
  if [ -z "$JOBID" ]; then
    echo "[chain] no JOBID parsed; retrying in 60s" | tee -a "$JOURNAL"
    sleep 60; continue
  fi
  echo "[chain] jobid=$JOBID" | tee -a "$JOURNAL"

  # --- resolve master address for torchrun rendezvous ------------------------
  MASTER_NODE=$(scontrol show hostname $(squeue -h -j $JOBID -o "%N") 2>/dev/null | head -1)
  if [ -z "$MASTER_NODE" ]; then
    echo "[chain] could not resolve MASTER_NODE; cancelling and retrying" | tee -a "$JOURNAL"
    scancel $JOBID; sleep 30; continue
  fi

  # --- decide resume --------------------------------------------------------
  RESUME_ARGS=""
  if [ -f "$OUT_DIR/last.ckpt" ]; then
    RESUME_ARGS="--resume_from $OUT_DIR"
    echo "[chain] will resume from $OUT_DIR/last.ckpt" | tee -a "$JOURNAL"
  else
    echo "[chain] no last.ckpt yet; starting fresh" | tee -a "$JOURNAL"
  fi

  TRAIN_LOG=$LOGDIR/train_${JOBID}.log

  # --- launch via torchrun-under-srun (same pattern as overnight_1M_ddp) ----
  srun --jobid=$JOBID --nodes=4 --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-node=4 \
    bash -c "cd $REPO && unset TMPDIR && \
      WANDB__SERVICE_WAIT=300 NCCL_SOCKET_IFNAME=\"^lo,docker0\" MASTER_ADDR=$MASTER_NODE MASTER_PORT=29500 \
      $PYTHON -m torch.distributed.run \
        --nproc_per_node=4 --nnodes=4 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_NODE:29500 \
        --node_rank=\$SLURM_NODEID \
        -m nano_hep.train_hep --config $CFG $RESUME_ARGS" \
    > "$TRAIN_LOG" 2>&1
  SRUN_EC=$?
  echo "[chain] srun exit=$SRUN_EC  (log: $TRAIN_LOG)" | tee -a "$JOURNAL"

  scancel $JOBID 2>/dev/null

  if [ $SRUN_EC -eq 0 ]; then
    echo "[chain] clean exit → training finished. Stopping chain." | tee -a "$JOURNAL"
    break
  fi
  echo "[chain] non-clean exit ($SRUN_EC). Likely SLURM timeout; continuing." | tee -a "$JOURNAL"
  sleep 15
done

echo "[chain] DONE @ $(date). out_dir=$OUT_DIR" | tee -a "$JOURNAL"
