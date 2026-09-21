#!/usr/bin/env bash
set -e  # stop on the first failure, rather than silently continuing

# Same --epochs for every run, so the comparison is fair.
EPOCHS=25

run() {
    local data_dir=$1
    local dest=$2
    echo ">>> Training on data/$data_dir -> results/$dest"
    python main.py --dataset SEGTHOR --data_dir "$data_dir" --dest "results/$dest" \
        --epochs $EPOCHS --gpu
}

# run SEGTHOR_none               none
# run SEGTHOR_clip               clip
run SEGTHOR_clip_resample      clip_resample
run SEGTHOR_resample           resample
run SEGTHOR_normalize          normalize
run SEGTHOR_clip_normalize     clip_normalize
run SEGTHOR_resample_normalize resample_normalize
run SEGTHOR_full               full

echo ">>> All 8 training runs done."