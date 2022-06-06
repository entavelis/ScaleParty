#!/usr/bin/env bash

PROGRAM="$1"
CKPT="checkpoints/$2/ckpt/latest.pth"
CONFIG="configs/scaleparty/$2.py"
PY_ARGS=${@:3}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -u tools/evaluation.py ${CONFIG} ${CKPT} --launcher="none" ${PY_ARGS}
echo "python $PROGRAM $CONFIG $CKPT $PY_ARGS"
python $PROGRAM $CONFIG $CKPT $PY_ARGS
