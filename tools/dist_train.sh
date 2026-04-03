#!/usr/bin/env bash
# ---------------------------------------------------------------
# Distributed training launcher
# Usage: ./tools/dist_train.sh <CONFIG> <NUM_GPUS> [PY_ARGS]
#
# Examples:
#   ./tools/dist_train.sh configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py 4
#   ./tools/dist_train.sh configs/segformer/segformer_mit-b5_openearthmap_train1000_40k.py 2 --work-dir work_dirs/exp1
# ---------------------------------------------------------------

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch \
    ${@:3}
