#!/bin/bash

# on the VU compute we have access to 128 kernels; every worker starts number of kernels many threads, so with 5 workers we are roughly at 600+ threads;
# to avoid spending ressources on thread management, we limit the number of threads to 1 per worker; this is also recommended by pytorch and mkl
# now we increase the number of workers to 12 since they are cheap; see main.py

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJECT="/local/data/ipv577/projects/ai4mi_project_group13"
source "$PROJECT/ai4mi_venv/bin/activate"
cd "$PROJECT" || exit 1

SPLIT_BASE="$PROJECT/splits"
RESULTS_BASE="$PROJECT/results/context_size_experiment"

SEEDS=(42 43 44)
CONTEXT_SIZES=(1 3 5 7 9)
FOLD=0
EPOCHS=25

mkdir -p "$RESULTS_BASE"

for seed in "${SEEDS[@]}"; do
    DATA_DIR="$SPLIT_BASE/SEGTHOR_full_split${seed}"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "!!! missing $DATA_DIR -- slice it first, skipping split $seed"
        continue
    fi
 
    for ctx in "${CONTEXT_SIZES[@]}"; do
        # layout must match plot_context_results.py: <split>_train<seed>/context<n>
        DEST="$RESULTS_BASE/split${seed}_train${seed}/context${ctx}"

        # swapped the hole body of the loop so we now write after every successfull iteration to a file and can check that file in the next experiment in case it crashes
        # here we check whether the current configuration was ran already; if, we continue
        if [ -f "$DEST/done.txt" ]; then
            echo ">>> skip  split=$seed ctx=$ctx (done)"
            continue
        fi

        # ! after some experiments I figures out that some asserts in utils.py increase runtime by roughly 60% within the metrics compution;
        # ! therefore we run with flag -O to exclude all assertions
        echo "=== start split=$seed ctx=$ctx  $(date) ==="
        python -O main.py \
            --dataset SEGTHOR \
            --mode full \
            --epochs "$EPOCHS" \
            --dest "$DEST" \
            --data_dir "$DATA_DIR" \
            --context_size "$ctx" \
            --seed "$seed" \
            --gpu

        # here we write to the done.txt that current configuration was ran successfully
        if [ $? -eq 0 ]; then
            touch "$DEST/done.txt"
            echo "=== done  split=$seed ctx=$ctx  $(date) ==="
        else
            echo "!!! FAILED split=$seed ctx=$ctx  $(date)"
        fi
    done
done    

echo "### all runs attempted $(date)"
echo "### finished:"
find "$RESULTS_BASE" -name done.txt | sed "s|$RESULTS_BASE/||;s|/done.txt||" | sort