#!/usr/bin/env bash
set -e  # stop on the first failure, rather than silently continuing

SRC="data/mostly_corrected_data"
RETAINS=4

echo ">>> 0. none (true baseline)"
python slice_segthor.py --source_dir "$SRC" --dest_dir data/SEGTHOR_none --retains $RETAINS

echo ">>> 1. clip only"
python slice_segthor.py --source_dir "$SRC" --dest_dir data/SEGTHOR_clip --retains $RETAINS --clip

echo ">>> 2. clip + resample"
python slice_segthor.py --source_dir "$SRC" --dest_dir data/SEGTHOR_clip_resample --retains $RETAINS --clip --resample

echo ">>> 3. resample only"
python slice_segthor.py --source_dir "$SRC" --dest_dir data/SEGTHOR_resample --retains $RETAINS --resample

echo ">>> 4. normalize only"
python slice_segthor.py --source_dir "$SRC" --dest_dir data/SEGTHOR_normalize --retains $RETAINS --normalize

echo ">>> 5. clip + normalize"
python slice_segthor.py --source_dir "$SRC" --dest_dir data/SEGTHOR_clip_normalize --retains $RETAINS --clip --normalize

echo ">>> 6. resample + normalize"
python slice_segthor.py --source_dir "$SRC" --dest_dir data/SEGTHOR_resample_normalize --retains $RETAINS --resample --normalize

echo ">>> 7. clip + resample + normalize (full pipeline)"
python slice_segthor.py --source_dir "$SRC" --dest_dir data/SEGTHOR_full --retains $RETAINS --clip --resample --normalize

echo ">>> All 8 done."