#!/bin/bash
set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1



RAW_DATA=/local/data/mjp104/ai4mi_segthor_clean



PROCESSED_ROOT_NO_Z=/local/data/mjp104/ai4mi_segthor_processed_spacing_no_resampling_physical_context
RESULTS_ROOT_NO_Z=/local/data/mjp104/ai4mi_results/spacing_no_resampling_physical_context


PROCESSED_ROOT_Z=/local/data/mjp104/ai4mi_segthor_processed_spacing_resampling_and_physical_context
RESULTS_ROOT_Z=/local/data/mjp104/ai4mi_results/spacing_resampling_and_physical_context



SEEDS=(44)
FOLD=0

CONTEXT_SIZES=(1 3 5 7 9)


mkdir -p "$PROCESSED_ROOT_NO_Z"
mkdir -p "$RESULTS_ROOT_NO_Z"

mkdir -p "$PROCESSED_ROOT_Z"
mkdir -p "$RESULTS_ROOT_Z"



for seed in "${SEEDS[@]}"; do


    SLICE_DIR_NO_Z="${PROCESSED_ROOT_NO_Z}/SEGTHOR_split${seed}_fold${FOLD}"

    if [ ! -d "$SLICE_DIR_NO_Z" ]; then


        python slice_segthor.py \
            --source_dir "$RAW_DATA" \
            --dest_dir "$SLICE_DIR_NO_Z" \
            --shape 512 512 \
            --retains 10 \
            --seed "$seed" \
            --fold "$FOLD" \
            --clip \
            --resample \
            --crop \
            --normalize

    else

        echo ""
        echo "NO Z-RESAMPLING preprocessing already exists:"
        echo "$SLICE_DIR_NO_Z"
        echo ""

    fi


    SLICE_DIR_Z="${PROCESSED_ROOT_Z}/SEGTHOR_split${seed}_fold${FOLD}"

    if [ ! -d "$SLICE_DIR_Z" ]; then


        python slice_segthor.py \
            --source_dir "$RAW_DATA" \
            --dest_dir "$SLICE_DIR_Z" \
            --shape 512 512 \
            --retains 10 \
            --seed "$seed" \
            --fold "$FOLD" \
            --clip \
            --resample \
            --resample_z \
            --crop \
            --normalize

    else

        echo ""
        echo "Z-RESAMPLING preprocessing already exists:"
        echo "$SLICE_DIR_Z"
        echo ""

    fi


    for ctx in "${CONTEXT_SIZES[@]}"; do


        DEST_NO_Z="${RESULTS_ROOT_NO_Z}/split${seed}_train${seed}/context${ctx}"


        python -O main.py \
            --dataset SEGTHOR \
            --mode full \
            --epochs 25 \
            --dest "$DEST_NO_Z" \
            --data_dir "$SLICE_DIR_NO_Z" \
            --context_size "$ctx" \
            --physical_context \
            --seed "$seed" \
            --gpu



        DEST_Z="${RESULTS_ROOT_Z}/split${seed}_train${seed}/context${ctx}"


        python -O main.py \
            --dataset SEGTHOR \
            --mode full \
            --epochs 25 \
            --dest "$DEST_Z" \
            --data_dir "$SLICE_DIR_Z" \
            --context_size "$ctx" \
            --physical_context \
            --seed "$seed" \
            --gpu

    done

done
