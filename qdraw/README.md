# QDRAW

[Quick Draw! Doodle Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition)

# Goal
- Experiment on different methods.
- Get a medal (failed).

# Final

Got public score 0.93273 and private score 0.93471 at 9-th cycle with the following setups (late submission, 7 days with one Nvidia V100, still no medal).

```
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

train-local: CKPT_PATH=./ckpt/$(MODEL)_$(TIMESTAMP)
train-local: LOGS_PATH=./logs/$(MODEL)_$(TIMESTAMP)
train-local: train

train-mobilenets: MODEL=mobilenets
train-mobilenets: TRAIN_ON_RECOGNIZED=false
train-mobilenets: IMAGE_SIZE=256
train-mobilenets: VALID_CYCLE=10000
train-mobilenets: OPTIMIZER=adam
train-mobilenets: CYCLIC_NUM_STEPS=36000
train-mobilenets: CYCLIC_NUM_CYCLES=30
train-mobilenets: CYCLIC_BATCH_SIZE=340
train-mobilenets: CYCLIC_BATCH_SIZE_MULTIPLIER_MIN=2
train-mobilenets: CYCLIC_BATCH_SIZE_MULTIPLIER_MAX=4
train-mobilenets: CYCLIC_LEARNING_RATE_MAX=0.0002
train-mobilenets: CYCLIC_LEARNING_RATE_MIN=0.0000004
train-mobilenets: SWA_ENABLE=false
train-mobilenets: TTA_ENABLE=false
train-mobilenets: SLR_NUM_TRIALS=0
train-mobilenets: SLR_NUM_STEPS=2000
train-mobilenets: SLR_RANDOM=false
train-mobilenets: SLR_BATCH_SIZE_MULTIPLIER=2
train-mobilenets: SLR_LEARNING_RATE_MIN=0.0000001
train-mobilenets: SLR_LEARNING_RATE_MAX=0.0000101
train-mobilenets: train-local
```

# References


# Note
- Try pre-built models in the beginning next time.
- The forum is a very good place to learn things, even you only read papers they refered.