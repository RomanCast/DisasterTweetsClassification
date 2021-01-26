# Domain Adaptation for Disaster Tweet Classification

This repo contains the code for the MVA Deep Learning course project 2020.

It contains code to train a sentence classifier on Disaster Tweet data. In order to make it work, you will need to get the dataset from [Alam et al., 2020](https://arxiv.org/abs/2004.06774), available [here](https://crisisnlp.qcri.org/crisis_datasets_benchmarks.html). All the labels are already in the folder `data` of this repository.

You can also train RoBERTa adversarially, following [Domain Adversarial training for Neural Networks](). This is useful to improve robustness of your model. The modified model is in the file `model.py`, and is mostly inspired by (https://github.com/bzantium/bert-DANN).

We offer the option to use [Weights and Biases](https://wandb.ai/) to log experiments. Add the flag `--use_wandb` to enable it.

## Run training of the original classifier

To train with specific sources, use the `--train_sources` argument with any number of sources you wish. If you want to train with different events, replace that argument by `--train_events` and specify any event from the dataset you would like.

```
python main.py --dataset_filename <TRAINING_FILENAME> --model_name_or_path roberta-base --train_sources <TRAIN_SOURCES> --n_epochs 10 --batch_size 32 --lr 1e-5
```

## Run training of the domain adversarial classifier

Training of the adversarial model is similar, except that you need to specify the unlabeled data (target) you want to use, using `--target_sources` or `--target_events`.

**Be careful :** only RoBERTa is available for this training.

```
python main_adversarial.py --dataset_filename <TRAINING_FILENAME> --train_sources <TRAIN_SOURCES> --target_sources <TARGET_SOURCES> --n_epochs 10 --batch_size 16 --lr 1e-5
```

## Run evaluation of the classifiers on new domains

Add the flag `--adversarial` to evaluate the adversarial model. In that case, add the target domain on which it was trained on using `--target_sources`.
```
python test.py --model_name_or_path roberta-base --train_filename <TRAINING_FILENAME> --test_filename <TRAINING_FILENAME> --train_sources <TRAIN_SOURCES> --test_sources $TARGET_SOURCES
```
