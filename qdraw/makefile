.PHONY:	deploy-worker report train

TIMESTAMP=$(shell date +%Y%m%d%H%M%S)

# general
PROJECT_NAME=qdraw
PROJECT_PATH=/home/ironhead/projects/ml_kaggle/$(PROJECT_NAME)/$(PROJECT_NAME)
DATASET_ROOT=/home/ironhead/datasets/$(PROJECT_NAME)
DATASET_PATH_DUMP=$(DATASET_ROOT)/dump
DATASET_PATH_RAW_TRAIN=$(DATASET_ROOT)/train

# worker
deploy-worker:
	gcloud compute scp $(PROJECT_NAME)/*.py $(WORKER_HOST):$(PROJECT_PATH) --zone=$(WORKER_ZONE)
	gcloud compute scp makefile $(WORKER_HOST):$(PROJECT_PATH)/../ --zone=$(WORKER_ZONE)

deploy-hecate: WORKER_HOST=hecate-worker
deploy-hecate: WORKER_ZONE=asia-east1-b
deploy-hecate: deploy-worker

deploy-hecate-gpu: WORKER_HOST=hecate-worker-tesla-p100
deploy-hecate-gpu: WORKER_ZONE=asia-east1-a
deploy-hecate-gpu: deploy-worker

train:
	python3 -m qdraw.experiment_train \
		--ckpt_path=$(CKPT_PATH)/ \
		--logs_path=$(LOGS_PATH)/ \
		--model=$(MODEL) \
		--train_dir_path=/home/ironhead/datasets/qdraw/28x28/train_pure/ \
		--valid_tfr_path=/home/ironhead/datasets/qdraw/28x28/valid/valid.tfrecord \
		--image_size=28 \
		--batch_size=100 \
		--stop_at_step=500000
	python3 -m qdraw.experiment_test \
		--ckpt_path=$(CKPT_PATH)/model.ckpt \
		--source_path=/home/ironhead/datasets/qdraw/28x28/test/test_simplified_28x28.tfrecord \
		--result_path=./results_npz/$(MODEL)_$(TIMESTAMP).npz
	python3 -m qdraw.experiment_finalize \
		--source_npz_path=./results_npz/$(MODEL)_$(TIMESTAMP).npz \
		--source_csv_path=/home/ironhead/datasets/qdraw/test_simplified.csv \
		--result_zip_path=./results_gz/$(MODEL)_$(TIMESTAMP).gz

train-hecate: CKPT_PATH=gs://hecate-research-ml-results/qdraw/ckpt/$(MODEL)_$(TIMESTAMP)
train-hecate: LOGS_PATH=gs://hecate-research-ml-results/qdraw/logs/$(MODEL)_$(TIMESTAMP)
train-hecate: train

train-hecate-cnn: MODEL=cnn
train-hecate-cnn: train-hecate

report:
	python -m qdraw.dataset_explorer --source_dir=$(DATASET_PATH_RAW_TRAIN) --result_dir=$(DATASET_PATH_DUMP)

explore_random:
	python -m qdraw.dataset_preprocessor --dry_run --output_type=png --source_dir=$(DATASET_PATH_RAW_TRAIN) --result_dir=./


test:
	python -m qdraw.experiment_test \
		--ckpt_path=gs://hecate-research-ml-results/qdraw/ckpt/cnn_20181010091812/model.ckpt \
		--source_path=/home/ironhead/datasets/qdraw/28x28/test/test_simplified_28x28.tfrecord \
		--result_path=xxx.npz

