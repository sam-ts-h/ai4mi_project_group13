#!/bin/bash
set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

RAW_DATA=/local/data/mjp104/ai4mi_segthor_clean
PROCESSED_ROOT=/local/data/mjp104/ai4mi_segthor_processed
RESULTS_ROOT=/local/data/mjp104/ai4mi_results
SEEDS=(42 43 44)
FOLD=0
CONTEXT_SIZES=(1 5)

mkdir -p "$PROCESSED_ROOT" "$RESULTS_ROOT"

for seed in "${SEEDS[@]}"; do
    SLICE_DIR="${PROCESSED_ROOT}/SEGTHOR_split${seed}_fold${FOLD}"

    if [ ! -d "$SLICE_DIR" ]; then
        python slice_segthor.py --source_dir "$RAW_DATA" --dest_dir "$SLICE_DIR" \
            --shape 544 352 --retains 10 --seed "$seed" --fold "$FOLD" \
            --clip --resample --resample_z --crop --normalize
    fi

    for ctx in "${CONTEXT_SIZES[@]}"; do
        prefix=$([ "$ctx" -eq 1 ] && echo "C01" || echo "C02")
        dest="${RESULTS_ROOT}/z_resample_experiment/split${seed}_train${seed}/${prefix}_context${ctx}"

        python -O main.py --dataset SEGTHOR --mode full --epochs 25 \
            --dest "$dest" --data_dir "$SLICE_DIR" \
            --context_size "$ctx" --seed "$seed" --gpu
    done
done