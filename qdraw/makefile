.PHONY:	deploy-worker report train

TIMESTAMP=$(shell date +%Y%m%d%H%M%S)

# general
PROJECT_NAME=qdraw
PROJECT_PATH=/home/ironhead/projects/ml_kaggle/$(PROJECT_NAME)/$(PROJECT_NAME)
DATASET_ROOT=/home/ironhead/datasets/$(PROJECT_NAME)
DATASET_PATH_DUMP=$(DATASET_ROOT)/dump
DATASET_PATH_RAW_TRAIN=$(DATASET_ROOT)/train

# worker
deploy-gcp:
	gcloud compute scp $(PROJECT_NAME)/*.py $(WORKER_HOST):$(PROJECT_PATH) --zone=$(WORKER_ZONE)
	gcloud compute scp makefile $(WORKER_HOST):$(PROJECT_PATH)/../ --zone=$(WORKER_ZONE)

deploy-hecate: WORKER_HOST=hecate-worker
deploy-hecate: WORKER_ZONE=asia-east1-a
deploy-hecate: deploy-gcp

deploy-hecate-gpu: WORKER_HOST=hecate-worker-tesla-p100
deploy-hecate-gpu: WORKER_ZONE=asia-east1-a
deploy-hecate-gpu: deploy-gcp

deploy-local:
	scp $(PROJECT_NAME)/*.py $(WORKER_HOST):$(PROJECT_PATH)
	scp makefile $(WORKER_HOST):$(PROJECT_PATH)/../

deploy-karina: WORKER_HOST=192.168.0.16
deploy-karina: deploy-local

deploy-laurana: WORKER_HOST=172.30.67.24
deploy-laurana: deploy-local

train:
	python3 -m qdraw.experiment_train \
		--ckpt_path=$(CKPT_PATH)/ \
		--logs_path=$(LOGS_PATH)/ \
		--train_dir_path=/home/ironhead/datasets/qdraw/complex/train/ \
		--valid_tfr_path=/home/ironhead/datasets/qdraw/complex/valid/origin_0000.tfrecord \
		--test_tfr_path=/home/ironhead/datasets/qdraw/complex/test/test.tfrecord \
		--result_zip_path=./results_gz/$(MODEL)_$(TIMESTAMP).gz \
		--model=$(MODEL) \
		--save_checkpoint=false \
		--image_size=32 \
		--batch_size=$(BATCH_SIZE) \
		--initial_learning_rate=$(INITIAL_LEARNING_RATE) \
		--stop_at_step=200000

train-hecate: CKPT_PATH=gs://hecate-research-ml-results/qdraw/ckpt/$(MODEL)_$(TIMESTAMP)
train-hecate: LOGS_PATH=gs://hecate-research-ml-results/qdraw/logs/$(MODEL)_$(TIMESTAMP)
train-hecate: BATCH_SIZE=512
train-hecate: train

train-hecate-cnn: INITIAL_LEARNING_RATE=0.0001
train-hecate-cnn: MODEL=cnn
train-hecate-cnn: train-hecate

train-hecate-rnn: INITIAL_LEARNING_RATE=0.0001
train-hecate-rnn: MODEL=rnn
train-hecate-rnn: train-hecate

train-hecate-mobilenets: INITIAL_LEARNING_RATE=0.0002
train-hecate-mobilenets: MODEL=mobilenets
train-hecate-mobilenets: train-hecate

train-local: BATCH_SIZE=100
train-local: CKPT_PATH=./ckpt/$(MODEL)_$(TIMESTAMP)
train-local: LOGS_PATH=./logs/$(MODEL)_$(TIMESTAMP)
train-local: train

train-local-rnn: INITIAL_LEARNING_RATE=0.0001
train-local-rnn: MODEL=rnn
train-local-rnn: train-local

train-local-cnn: INITIAL_LEARNING_RATE=0.0001
train-local-cnn: MODEL=cnn
train-local-cnn: train-local

train-local-mobilenets: INITIAL_LEARNING_RATE=0.002
train-local-mobilenets: MODEL=mobilenets
train-local-mobilenets: train-local

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

preprocess-test:
	python3 -m qdraw.dataset_preprocessor \
		--add_perturbation=false \
		--image_size=32 \
		--index_label=-1 \
		--index_keyid=0 \
		--index_strokes=2 \
		--prefix=test \
		--source_csv_path=/home/ironhead/datasets/qdraw/kaggle/test_simplified.csv \
		--result_tfr_path=/home/ironhead/datasets/qdraw/complex/test/test.tfrecord

preprocess-valid:
	python3 -m qdraw.dataset_preprocessor \
		--add_perturbation=false \
		--image_size=32 \
		--index_label=5 \
		--index_keyid=2 \
		--index_strokes=1 \
		--prefix=origin \
		--source_dir=/home/ironhead/datasets/qdraw/complex/split_valid/ \
		--result_dir=/home/ironhead/datasets/qdraw/complex/valid/

preprocess-train:
	python3 -m qdraw.dataset_preprocessor \
		--add_perturbation=false \
		--image_size=32 \
		--index_label=5 \
		--index_keyid=2 \
		--index_strokes=1 \
		--prefix=origin \
		--source_dir=/home/ironhead/datasets/qdraw/complex/split_train/ \
		--result_dir=/home/ironhead/datasets/qdraw/complex/train/

