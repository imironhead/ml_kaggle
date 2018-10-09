.PHONY:	deploy-worker report train

TIMESTAMP=$(shell date +%Y%m%d%H%M%S)

# general
PROJECT_NAME=qdraw
PROJECT_PATH=/home/ironhead/projects/ml_kaggle/$(PROJECT_NAME)/$(PROJECT_NAME)
DATASET_ROOT=/home/ironhead/datasets/$(PROJECT_NAME)
DATASET_PATH_DUMP=$(DATASET_ROOT)/dump
DATASET_PATH_RAW_TRAIN=$(DATASET_ROOT)/train

# worker
WORKER_HOST=hecate-worker-tesla-p100
WORKER_ZONE=asia-east1-a

deploy-worker:
	gcloud compute scp $(PROJECT_NAME)/*.py $(WORKER_HOST):$(PROJECT_PATH) --zone=$(WORKER_ZONE)
	gcloud compute scp makefile $(WORKER_HOST):$(PROJECT_PATH)/../ --zone=$(WORKER_ZONE)

report:
	python -m qdraw.dataset_explorer --source_dir=$(DATASET_PATH_RAW_TRAIN) --result_dir=$(DATASET_PATH_DUMP)

explore_random:
	python -m qdraw.dataset_preprocessor --dry_run --output_type=png --source_dir=$(DATASET_PATH_RAW_TRAIN) --result_dir=./

train:
	python -m qdraw.experiment_train \
		--ckpt_path=gs://hecate-research-ml-results/qdraw/ckpt/cnn_$(TIMESTAMP) \
		--logs_path=gs://hecate-research-ml-results/qdraw/logs/cnn_$(TIMESTAMP) \
		--model=cnn \
		--train_dir_path=/home/ironhead/datasets/qdraw/28x28/train_pure/ \
		--valid_tfr_path=/home/ironhead/datasets/qdraw/28x28/valid/valid.tfrecord \
		--image_size=28 \
		--batch_size=100

