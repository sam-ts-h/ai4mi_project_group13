# on the VU compute we have access to 128 kernels; every worker starts number of kernels many threads, so with 5 workers we are roughly at 600+ threads;
# to avoid spending ressources on thread management, we limit the number of threads to 1 per worker; this is also recommended by pytorch and mkl
# now we increase the number of workers to 12 since they are cheap; see main.py

#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJECT="/local/data/ipv577/projects/ai4mi_project_group13"
source "$PROJECT/ai4mi_venv/bin/activate"
cd "$PROJECT" || exit 1

# set after sourcing the venv, some activate scripts touch unset variables
set -u   # abort on undefined variables, would have caught $DEST and $ctx

SPLIT_BASE="$PROJECT/splits"
RESULTS_BASE="$PROJECT/results/2d_adamW"

SEEDS=(42 43 44)
FOLD=0
EPOCHS=25

mkdir -p "$RESULTS_BASE"

for seed in "${SEEDS[@]}"; do
    DATA_DIR="$SPLIT_BASE/SEGTHOR_full_split${seed}"
    DEST="$RESULTS_BASE/seed${seed}"   # one result dir per seed

    if [ ! -d "$DATA_DIR" ]; then
        echo "!!! missing $DATA_DIR -- slice it first, skipping seed $seed"
        continue
    fi

    # skip configurations that already finished successfully
    if [ -f "$DEST/done.txt" ]; then
        echo ">>> skip seed=$seed (done)"
        continue
    fi

    mkdir -p "$DEST"
    # log which code version produced these results
    git rev-parse HEAD > "$DEST/commit.txt"
    git diff --quiet || echo "!!! warning: uncommitted changes, not captured by commit.txt"

    echo "=== start seed=$seed  $(date) ==="
    # ! after some experiments I figures out that some asserts in utils.py increase runtime by roughly 60% within the metrics compution;
    # ! therefore we run with flag -O to exclude all assertions
    python -O main.py \
        --dataset SEGTHOR \
        --mode full \
        --epochs "$EPOCHS" \
        --dest "$DEST" \
        --data_dir "$DATA_DIR" \
        --seed "$seed" \
        --gpu \

    if [ $? -eq 0 ]; then
        touch "$DEST/done.txt"
        echo "=== done seed=$seed  $(date) ==="
    else
        echo "!!! FAILED seed=$seed  $(date)"
    fi
done

echo "### all runs attempted $(date)"
echo "### finished:"
find "$RESULTS_BASE" -name done.txt | sed "s|$RESULTS_BASE/||;s|/done.txt||" | sort