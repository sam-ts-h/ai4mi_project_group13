#!/bin/bash
set -e

source /local/data/ipv577/projects/ai4mi_project_group13/ai4mi_venv/bin/activate

SPLIT_BASE="/local/data/ipv577/projects/ai4mi_project_group13/splits"
RESULTS_BASE="/local/data/ipv577/projects/ai4mi_project_group13/results"

SEEDS=(42 43 44)
CONTEXT_SIZES=(1 3 5 7 9)
FOLD=0
EPOCHS=25

mkdir -p "$RESULTS_BASE"

for seed in "${SEEDS[@]}"; do
    DATA_DIR="$SPLIT_BASE/SEGTHOR_full_split${seed}"

    for ctx in "${CONTEXT_SIZES[@]}"; do
        DEST="$RESULTS_BASE/split${seed}_train${seed}/fold${FOLD}/context${ctx}"

        echo "=========================================="
        echo "Starting: split=${seed}, train_seed=${seed}, fold=${FOLD}, context=${ctx}"
        echo "=========================================="

        python main.py \
            --dataset SEGTHOR \
            --mode full \
            --epochs "$EPOCHS" \
            --dest "$DEST" \
            --data_dir "$DATA_DIR" \
            --context_size "$ctx" \
            --seed "$seed" \
            --gpu
    done
done

echo "=========================================="
echo "ALL 15 RUNS COMPLETED"
echo "=========================================="
