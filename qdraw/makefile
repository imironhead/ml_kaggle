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
		--model=$(MODEL) \
		--train_dir_path=/home/ironhead/datasets/qdraw/complex/train/ \
		--valid_tfr_path=/home/ironhead/datasets/qdraw/complex/valid/_0000.tfrecord \
		--image_size=32 \
		--batch_size=100 \
		--stop_at_step=500000
	python3 -m qdraw.experiment_test \
		--image_size=32 \
		--ckpt_path=$(CKPT_PATH)/model.ckpt \
		--source_path=/home/ironhead/datasets/qdraw/complex/test/test.tfrecord \
		--result_path=./results_npz/$(MODEL)_$(TIMESTAMP).npz
	python3 -m qdraw.experiment_finalize \
		--source_npz_path=./results_npz/$(MODEL)_$(TIMESTAMP).npz \
		--source_csv_path=/home/ironhead/datasets/qdraw/kaggle/test_simplified.csv \
		--result_zip_path=./results_gz/$(MODEL)_$(TIMESTAMP).gz

train-hecate: CKPT_PATH=gs://hecate-research-ml-results/qdraw/ckpt/$(MODEL)_$(TIMESTAMP)
train-hecate: LOGS_PATH=gs://hecate-research-ml-results/qdraw/logs/$(MODEL)_$(TIMESTAMP)
train-hecate: train

train-hecate-cnn: MODEL=cnn
train-hecate-cnn: train-hecate

train-local: CKPT_PATH=./ckpt/$(MODEL)_$(TIMESTAMP)
train-local: LOGS_PATH=./logs/$(MODEL)_$(TIMESTAMP)
train-local: train

train-karina-resnet: MODEL=resnet
train-karina-resnet: train-local

train-karina-cnn: MODEL=cnn
train-karina-cnn: train-local

train-laurana-cnn: MODEL=cnn
train-laurana-cnn: train-local

report:
	python -m qdraw.dataset_explorer --source_dir=$(DATASET_PATH_RAW_TRAIN) --result_dir=$(DATASET_PATH_DUMP)

explore_random:
	python -m qdraw.dataset_preprocessor --dry_run --output_type=png --source_dir=$(DATASET_PATH_RAW_TRAIN) --result_dir=./



