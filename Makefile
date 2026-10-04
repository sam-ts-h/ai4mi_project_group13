red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor
## Original one
data/segthor_part1: data/segthor_part1.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_part1.sha256
	unzip -q $<
	rm -f $@/.DS_STORE

data/SEGTHOR:
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/segthor_part1 --dest_dir $@_tmp \
		--retain 5 --clip --resample --normalize --distmap
	mv $@_tmp $@

outputFile = experiments/exp_cedice
# ce of ceDice of ceDiceBoundary 
lossFn = ceDice

trainData: data/SEGTHOR
	python main.py --dataset SEGTHOR --mode full --loss $(lossFn) --epochs 25 --dest $(outputFile) --gpu


evalData:
	python stitch.py --data_folder $(outputFile)/best_epoch/val --dest_folder $(outputFile)/volumes \
		--num_classes 255 --grp_regex "(Patient_\\d\\d)_\\d\\d\\d\\d" \
		--source_scan_pattern "data/segthor_train_full/train/{id_}/GT.nii.gz" \
		--crop_centers data/SEGTHOR/crop_centers.pkl
	python metrics.py --pred_folder $(outputFile)/volumes --gt_folder data/segthor_train_full/train \
		--dest_folder $(outputFile)/metrics
	python plot.py --metric_file $(outputFile)/dice_val.npy --dest $(outputFile)/dice_val.png --headless
	python plot.py --metric_file $(outputFile)/loss_val.npy --dest $(outputFile)/loss_val.png --headless
	python plot.py --metric_file $(outputFile)/loss_tra.npy --dest $(outputFile)/loss_tra.png --headless
