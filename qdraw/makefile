.PHONY:	deploy-worker report train

TIMESTAMP=$(shell date +%Y%m%d%H%M%S)

# general
PROJECT_NAME=qdraw
PROJECT_PATH=/home/ironhead/projects/ml_kaggle/$(PROJECT_NAME)/$(PROJECT_NAME)
DATASET_ROOT=/home/ironhead/datasets/$(PROJECT_NAME)

# worker
deploy-gcp: WORKER_ZONE=asia-east1-c
deploy-gcp:
	gcloud compute scp $(PROJECT_NAME)/*.py $(WORKER_HOST):$(PROJECT_PATH) --zone=$(WORKER_ZONE)
	gcloud compute scp makefile $(WORKER_HOST):$(PROJECT_PATH)/../ --zone=$(WORKER_ZONE)

deploy-gcp-p100: WORKER_HOST=hecate-worker-tesla-p100
deploy-gcp-p100: deploy-gcp

deploy-local:
	scp $(PROJECT_NAME)/*.py $(WORKER_HOST):$(PROJECT_PATH)
	scp makefile $(WORKER_HOST):$(PROJECT_PATH)/../

deploy-karina: WORKER_HOST=192.168.0.16
deploy-karina: deploy-local

deploy-laurana: WORKER_HOST=172.30.67.24
deploy-laurana: deploy-local

deploy-dev: WORKER_HOST=dev.kkstream.tv
deploy-dev:
	scp $(PROJECT_NAME)/*.py $(WORKER_HOST):/home/ironheadchuang/projects/ml_kaggle/qdraw/qdraw/
	scp makefile $(WORKER_HOST):/home/ironheadchuang/projects/ml_kaggle/qdraw/

train:
	python3 -m qdraw.experiment_train \
		--model=$(MODEL) \
		--ckpt_path=$(CKPT_PATH)/ \
		--logs_path=$(LOGS_PATH)/ \
		--train_dir_path=/home/ironhead/datasets/qdraw/complex/train_$(IMAGE_SIZE)/ \
		--valid_dir_path=/home/ironhead/datasets/qdraw/complex/valid_$(IMAGE_SIZE)/ \
		--test_dir_path=/home/ironhead/datasets/qdraw/complex/test_$(IMAGE_SIZE)/ \
		--result_zip_path=./results_gz/$(MODEL)_$(TIMESTAMP).gz \
		--valid_cycle=$(VALID_CYCLE) \
		--train_on_recognized=$(TRAIN_ON_RECOGNIZED) \
		--image_size=$(IMAGE_SIZE) \
		--optimizer=$(OPTIMIZER) \
		--cyclic_num_steps=$(CYCLIC_NUM_STEPS) \
		--cyclic_num_cycles=$(CYCLIC_NUM_CYCLES) \
		--cyclic_batch_size=$(CYCLIC_BATCH_SIZE) \
		--cyclic_batch_size_multiplier_min=$(CYCLIC_BATCH_SIZE_MULTIPLIER_MIN) \
		--cyclic_batch_size_multiplier_max=$(CYCLIC_BATCH_SIZE_MULTIPLIER_MAX) \
		--cyclic_learning_rate_min=$(CYCLIC_LEARNING_RATE_MIN) \
		--cyclic_learning_rate_max=$(CYCLIC_LEARNING_RATE_MAX) \
		--swa_enable=$(SWA_ENABLE) \
		--tta_enable=$(TTA_ENABLE) \
		--tta_num_samples_valid=34000 \
		--tta_num_samples_test=112199 \
		--slr_num_trials=$(SLR_NUM_TRIALS) \
		--slr_num_steps=$(SLR_NUM_STEPS) \
		--slr_random=$(SLR_RANDOM) \
		--slr_batch_size_multiplier=$(SLR_BATCH_SIZE_MULTIPLIER) \
		--slr_learning_rate_min=$(SLR_LEARNING_RATE_MIN) \
		--slr_learning_rate_max=$(SLR_LEARNING_RATE_MAX)


train-gcp: CKPT_PATH=gs://hecate-research-ml-results/qdraw/ckpt/$(MODEL)_$(TIMESTAMP)
train-gcp: LOGS_PATH=gs://hecate-research-ml-results/qdraw/logs/$(MODEL)_$(TIMESTAMP)
train-gcp: train


train-gcp-resnet: MODEL=resnet
train-gcp-resnet: TRAIN_ON_RECOGNIZED=false
train-gcp-resnet: IMAGE_SIZE=256
train-gcp-resnet: VALID_CYCLE=340000
train-gcp-resnet: OPTIMIZER=nesterov
train-gcp-resnet: CYCLIC_NUM_STEPS=340000
train-gcp-resnet: CYCLIC_NUM_CYCLES=3
train-gcp-resnet: CYCLIC_BATCH_SIZE=340
train-gcp-resnet: CYCLIC_BATCH_SIZE_MULTIPLIER_HEAD=2
train-gcp-resnet: CYCLIC_BATCH_SIZE_MULTIPLIER_TAIL=2
train-gcp-resnet: CYCLIC_LEARNING_RATE_HEAD=0.00075
train-gcp-resnet: CYCLIC_LEARNING_RATE_TAIL=0.000015
train-gcp-resnet: SWA_ENABLE=true
train-gcp-resnet: TTA_ENABLE=false
train-gcp-resnet: SLR_NUM_TRIALS=10
train-gcp-resnet: SLR_NUM_STEPS=2000
train-gcp-resnet: SLR_RANDOM=false
train-gcp-resnet: SLR_BATCH_SIZE_MULTIPLIER=1
train-gcp-resnet: SLR_LEARNING_RATE_MIN=0.000001
train-gcp-resnet: SLR_LEARNING_RATE_MAX=0.001
train-gcp-resnet: train-local


train-gcp-mobilenets: MODEL=mobilenets
train-gcp-mobilenets: TRAIN_ON_RECOGNIZED=false
train-gcp-mobilenets: IMAGE_SIZE=256
train-gcp-mobilenets: VALID_CYCLE=340000
train-gcp-mobilenets: OPTIMIZER=nesterov
train-gcp-mobilenets: CYCLIC_NUM_STEPS=340000
train-gcp-mobilenets: CYCLIC_NUM_CYCLES=3
train-gcp-mobilenets: CYCLIC_BATCH_SIZE=340
train-gcp-mobilenets: CYCLIC_BATCH_SIZE_MULTIPLIER_HEAD=2
train-gcp-mobilenets: CYCLIC_BATCH_SIZE_MULTIPLIER_TAIL=2
train-gcp-mobilenets: CYCLIC_LEARNING_RATE_HEAD=0.00075
train-gcp-mobilenets: CYCLIC_LEARNING_RATE_TAIL=0.000015
train-gcp-mobilenets: SWA_ENABLE=false
train-gcp-mobilenets: TTA_ENABLE=false
train-gcp-mobilenets: SLR_NUM_TRIALS=10
train-gcp-mobilenets: SLR_NUM_STEPS=2000
train-gcp-mobilenets: SLR_RANDOM=false
train-gcp-mobilenets: SLR_BATCH_SIZE_MULTIPLIER=2
train-gcp-mobilenets: SLR_LEARNING_RATE_MIN=0.000001
train-gcp-mobilenets: SLR_LEARNING_RATE_MAX=0.001
train-gcp-mobilenets: train-local


train-gcp-mobilenets-v2: MODEL=mobilenets_v2
train-gcp-mobilenets-v2: TRAIN_ON_RECOGNIZED=false
train-gcp-mobilenets-v2: IMAGE_SIZE=256
train-gcp-mobilenets-v2: VALID_CYCLE=340000
train-gcp-mobilenets-v2: OPTIMIZER=adam
train-gcp-mobilenets-v2: CYCLIC_NUM_STEPS=340000
train-gcp-mobilenets-v2: CYCLIC_NUM_CYCLES=3
train-gcp-mobilenets-v2: CYCLIC_BATCH_SIZE=170
train-gcp-mobilenets-v2: CYCLIC_BATCH_SIZE_MULTIPLIER_MIN=2
train-gcp-mobilenets-v2: CYCLIC_BATCH_SIZE_MULTIPLIER_MAX=4
train-gcp-mobilenets-v2: CYCLIC_LEARNING_RATE_MIN=0.00001
train-gcp-mobilenets-v2: CYCLIC_LEARNING_RATE_MAX=0.0008
train-gcp-mobilenets-v2: SWA_ENABLE=false
train-gcp-mobilenets-v2: TTA_ENABLE=false
train-gcp-mobilenets-v2: SLR_NUM_TRIALS=20
train-gcp-mobilenets-v2: SLR_NUM_STEPS=2000
train-gcp-mobilenets-v2: SLR_RANDOM=false
train-gcp-mobilenets-v2: SLR_BATCH_SIZE_MULTIPLIER=1
train-gcp-mobilenets-v2: SLR_LEARNING_RATE_MIN=0.001
train-gcp-mobilenets-v2: SLR_LEARNING_RATE_MAX=0.1
train-gcp-mobilenets-v2: train-local

#0.005 ~

train-local: CKPT_PATH=./ckpt/$(MODEL)_$(TIMESTAMP)
train-local: LOGS_PATH=./logs/$(MODEL)_$(TIMESTAMP)
train-local: train

train-local-null: MODEL=null
train-local-null: TRAIN_ON_RECOGNIZED=false
train-local-null: IMAGE_SIZE=64
train-local-null: VALID_CYCLE=1000
train-local-null: CYCLIC_NUM_STEPS=20000
train-local-null: CYCLIC_NUM_CYCLES=1
train-local-null: CYCLIC_BATCH_SIZE=10
train-local-null: CYCLIC_BATCH_SIZE_MULTIPLIER_HEAD=2
train-local-null: CYCLIC_BATCH_SIZE_MULTIPLIER_TAIL=4
train-local-null: CYCLIC_LEARNING_RATE_HEAD=0.002
train-local-null: CYCLIC_LEARNING_RATE_TAIL=0.00002
train-local-null: TTA_ENABLE=true
train-local-null: train-local


train-local-resnet: MODEL=resnet
train-local-resnet: TRAIN_ON_RECOGNIZED=false
train-local-resnet: IMAGE_SIZE=64
train-local-resnet: VALID_CYCLE=340000
train-local-resnet: OPTIMIZER=nesterov
train-local-resnet: CYCLIC_NUM_STEPS=340000
train-local-resnet: CYCLIC_NUM_CYCLES=3
train-local-resnet: CYCLIC_BATCH_SIZE=170
train-local-resnet: CYCLIC_BATCH_SIZE_MULTIPLIER_HEAD=2
train-local-resnet: CYCLIC_BATCH_SIZE_MULTIPLIER_TAIL=2
train-local-resnet: CYCLIC_LEARNING_RATE_HEAD=0.00075
train-local-resnet: CYCLIC_LEARNING_RATE_TAIL=0.000015
train-local-resnet: SWA_ENABLE=true
train-local-resnet: TTA_ENABLE=false
train-local-resnet: SLR_NUM_TRIALS=100
train-local-resnet: SLR_NUM_STEPS=2000
train-local-resnet: SLR_RANDOM=false
train-local-resnet: SLR_BATCH_SIZE_MULTIPLIER=1
train-local-resnet: SLR_LEARNING_RATE_MIN=0.000001
train-local-resnet: SLR_LEARNING_RATE_MAX=0.001
train-local-resnet: train-local


train-local-mobilenets: MODEL=mobilenets
train-local-mobilenets: TRAIN_ON_RECOGNIZED=false
train-local-mobilenets: IMAGE_SIZE=64
train-local-mobilenets: VALID_CYCLE=340000
train-local-mobilenets: OPTIMIZER=nesterov
train-local-mobilenets: CYCLIC_NUM_STEPS=340000
train-local-mobilenets: CYCLIC_NUM_CYCLES=3
train-local-mobilenets: CYCLIC_BATCH_SIZE=170
train-local-mobilenets: CYCLIC_BATCH_SIZE_MULTIPLIER_MIN=2
train-local-mobilenets: CYCLIC_BATCH_SIZE_MULTIPLIER_MAX=4
train-local-mobilenets: CYCLIC_LEARNING_RATE_MIN=0.00001
train-local-mobilenets: CYCLIC_LEARNING_RATE_MAX=0.0008
train-local-mobilenets: SWA_ENABLE=false
train-local-mobilenets: TTA_ENABLE=false
train-local-mobilenets: SLR_NUM_TRIALS=0
train-local-mobilenets: SLR_NUM_STEPS=10000
train-local-mobilenets: SLR_RANDOM=false
train-local-mobilenets: SLR_BATCH_SIZE_MULTIPLIER=2
train-local-mobilenets: SLR_LEARNING_RATE_MIN=0.00001
train-local-mobilenets: SLR_LEARNING_RATE_MAX=0.0008
train-local-mobilenets: train-local


train-local-mobilenets-v2: MODEL=mobilenets_v2
train-local-mobilenets-v2: TRAIN_ON_RECOGNIZED=false
train-local-mobilenets-v2: IMAGE_SIZE=64
train-local-mobilenets-v2: VALID_CYCLE=340000
train-local-mobilenets-v2: OPTIMIZER=nesterov
train-local-mobilenets-v2: CYCLIC_NUM_STEPS=340000
train-local-mobilenets-v2: CYCLIC_NUM_CYCLES=3
train-local-mobilenets-v2: CYCLIC_BATCH_SIZE=85
train-local-mobilenets-v2: CYCLIC_BATCH_SIZE_MULTIPLIER_HEAD=2
train-local-mobilenets-v2: CYCLIC_BATCH_SIZE_MULTIPLIER_TAIL=2
train-local-mobilenets-v2: CYCLIC_LEARNING_RATE_HEAD=0.00075
train-local-mobilenets-v2: CYCLIC_LEARNING_RATE_TAIL=0.000015
train-local-mobilenets-v2: SWA_ENABLE=false
train-local-mobilenets-v2: TTA_ENABLE=false
train-local-mobilenets-v2: SLR_NUM_TRIALS=20
train-local-mobilenets-v2: SLR_NUM_STEPS=10000
train-local-mobilenets-v2: SLR_RANDOM=true
train-local-mobilenets-v2: SLR_BATCH_SIZE_MULTIPLIER=4
train-local-mobilenets-v2: SLR_LEARNING_RATE_MIN=0.0000001
train-local-mobilenets-v2: SLR_LEARNING_RATE_MAX=0.01
train-local-mobilenets-v2: train-local

train-local-stochastic-depth: MODEL=stochastic_depth
train-local-stochastic-depth: IMAGE_SIZE=64
train-local-stochastic-depth: INITIAL_LEARNING_RATE=0.001
train-local-stochastic-depth: STOP_AT_STEP=500000
train-local-stochastic-depth: BATCH_SIZE=340
train-local-stochastic-depth: BATCH_MULTIPLIER=1
train-local-stochastic-depth: NUM_GPUS=1
train-local-stochastic-depth: TRAIN_ON_RECOGNIZED=false
train-local-stochastic-depth: train-local


train-local-blind: STOP_AT_STEP=200000
train-local-blind: TRAIN_ON_RECOGNIZED=false
train-local-blind: INITIAL_LEARNING_RATE=0.001
train-local-blind: MODEL=blind
train-local-blind: train-local

train-shallow: INITIAL_LEARNING_RATE=0.001
train-shallow: CKPT_PATH=./ckpt/$(MODEL)_$(TIMESTAMP)
train-shallow: LOGS_PATH=./logs/$(MODEL)_$(TIMESTAMP)
train-shallow: IMAGE_SIZE=64
train-shallow: BATCH_SIZE=640
train-shallow: TRAIN_ON_RECOGNIZED=false
train-shallow: MODEL=shallow
train-shallow:
	python3 -m qdraw.experiment_shallow \
		--ckpt_path=$(CKPT_PATH)/ \
		--logs_path=$(LOGS_PATH)/ \
		--train_dir_path=/home/ironhead/datasets/qdraw/complex/train_$(IMAGE_SIZE)/ \
		--valid_dir_path=/home/ironhead/datasets/qdraw/complex/valid_$(IMAGE_SIZE)/ \
		--test_dir_path=/home/ironhead/datasets/qdraw/complex/test_$(IMAGE_SIZE)/ \
		--result_zip_path=./results_gz/$(MODEL)_$(TIMESTAMP).gz \
		--model=$(MODEL) \
		--save_checkpoint=false \
		--image_size=$(IMAGE_SIZE) \
		--batch_size=$(BATCH_SIZE) \
		--initial_learning_rate=$(INITIAL_LEARNING_RATE) \
		--train_on_recognized=$(TRAIN_ON_RECOGNIZED) \
		--stop_at_step=200000

QDRAW_DATA_PATH=/home/ironhead/datasets/qdraw

# initial state
# $(PATH_DATASET)/kaggle/train_simplified.zip
# $(PATH_DATASET)/kaggle/test_simplified.csv
# to remove csv head:
# sed -i '1d' $(PATH_DATASET)/kaggle/test_simplified.csv
preprocess-prepare:
	mkdir $(QDRAW_DATA_PATH)/complex/
	mkdir $(QDRAW_DATA_PATH)/complex/split_train/
	mkdir $(QDRAW_DATA_PATH)/complex/split_valid/
	mkdir $(QDRAW_DATA_PATH)/complex/train/
	mkdir $(QDRAW_DATA_PATH)/complex/valid/
	mkdir $(QDRAW_DATA_PATH)/complex/test/
	unzip $(QDRAW_DATA_PATH)/kaggle/train_simplified.zip -d $(QDRAW_DATA_PATH)/complex/split_train/
	sed -i '1d' $(QDRAW_DATA_PATH)/complex/split_train/*
	rename 's/ /_/g' $(QDRAW_DATA_PATH)/complex/split_train/*
	ls $(QDRAW_DATA_PATH)/complex/split_train/ | sed 's/.csv//g' | xargs -I % mkdir $(QDRAW_DATA_PATH)/complex/split_train/%
	ls $(QDRAW_DATA_PATH)/complex/split_train/ | grep csv | sed 's/.csv//g' | xargs -I % mv $(QDRAW_DATA_PATH)/complex/split_train/%.csv $(QDRAW_DATA_PATH)/complex/split_train/%/
	ls $(QDRAW_DATA_PATH)/complex/split_train/ | xargs -P 32 -I % split -l 100 -d --suffix-length=8 --additional-suffix=.csv $(QDRAW_DATA_PATH)/complex/split_train/%/%.csv $(QDRAW_DATA_PATH)/complex/split_train/%/%-
	ls $(QDRAW_DATA_PATH)/complex/split_train/ | xargs -P 32 -I % rm $(QDRAW_DATA_PATH)/complex/split_train/%/%.csv
	ls $(QDRAW_DATA_PATH)/complex/split_train/ | xargs -P 32 -I % mkdir $(QDRAW_DATA_PATH)/complex/split_valid/%/
	ls $(QDRAW_DATA_PATH)/complex/split_train/ | xargs -P 32 -I % mv $(QDRAW_DATA_PATH)/complex/split_train/%/%-00000000.csv $(QDRAW_DATA_PATH)/complex/split_valid/%/
	ls $(QDRAW_DATA_PATH)/complex/split_train/ | xargs -P 32 -I % sh -c 'ls $(QDRAW_DATA_PATH)/complex/split_train/%/ | tail -n 1 | sed "s/^/%\//g"' | xargs -I % rm $(QDRAW_DATA_PATH)/complex/split_train/%

preprocess: IMAGE_SIZE=256
preprocess: PREFIX=perturb_cv2_0
preprocess: PERTURB=true
preprocess:
	mkdir -p /home/ironhead/datasets/qdraw/complex/test_$(IMAGE_SIZE)/
	mkdir -p /home/ironhead/datasets/qdraw/complex/valid_$(IMAGE_SIZE)/
	mkdir -p /home/ironhead/datasets/qdraw/complex/train_$(IMAGE_SIZE)/
	python3 -m qdraw.dataset_preprocess \
		--source_csv_path=/home/ironhead/datasets/qdraw/kaggle/test_simplified.csv \
		--result_tfr_path=/home/ironhead/datasets/qdraw/complex/test_$(IMAGE_SIZE)/$(PREFIX)_0000.tfrecord.gz \
		--prefix=$(PREFIX) \
		--perturb=$(PERTURB) \
		--image_size=$(IMAGE_SIZE) \
		--num_output=1 \
		--noshuffle
	python3 -m qdraw.dataset_preprocess \
		--source_dir=/home/ironhead/datasets/qdraw/complex/split_valid/ \
		--result_dir=/home/ironhead/datasets/qdraw/complex/valid_$(IMAGE_SIZE)/ \
		--prefix=$(PREFIX) \
		--perturb=$(PERTURB) \
		--image_size=$(IMAGE_SIZE) \
		--num_output=1 \
		--noshuffle
	python3 -m qdraw.dataset_preprocess \
		--source_dir=/home/ironhead/datasets/qdraw/complex/split_train/ \
		--result_dir=/home/ironhead/datasets/qdraw/complex/train_$(IMAGE_SIZE)/ \
		--prefix=$(PREFIX) \
		--perturb=$(PERTURB) \
		--image_size=$(IMAGE_SIZE) \
		--num_output=1000 \
		--shuffle

