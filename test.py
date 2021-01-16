import os
import json
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from sklearn.metrics import f1_score
import numpy as np

from data import HumanitarianDataset
from utils import get_model_name
from main import validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--train_filename", type=str, nargs='+', help="The dataset on which the model was trained on. Used to retrieve the labels.")
    parser.add_argument("--test_filename", type=str, nargs='+', help="The dataset on which the model will be evaluated.")
    parser.add_argument("--train_on", type=str, nargs="+", help="List of disaster on which the model was trained.")
    parser.add_argument("--test_on", type=str, nargs="+", help="Disaster on which to test the model.")
    parser.add_argument("--use_wandb", action='store_true')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.save_path = "models"

    # Get the model name on the existing folder containing models. If it does not exist, this will throw an exception
    args.finetuned_checkpoint = get_model_name(args, saving=False)

    # Get the number of unique labels
    unique_labels = json.load(open("data/humanitarian_labels.json", 'r'))
    num_labels = len(unique_labels)
    label_to_id = {l: i for i, l in enumerate(unique_labels)}

    # Load the tokenizer and model from HuggingFace Transformers. First time loading the model, it will download the pretrained weights.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )
    # We add the state_dict argument to load an existing finetuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
        state_dict=torch.load(os.path.join(args.save_path, args.finetuned_checkpoint))
    ).to(args.device)

    # We process each batch with this function. This way, we can pad to the specific length of every batch and avoid unnecessary long inputs
    def process_inputs(examples):
        inputs = tokenizer(examples["data"], padding=True, max_length=None, truncation=True)
        inputs = {key: torch.tensor(val) for key, val in inputs.items()}
        labels = torch.tensor([label_to_id[l] for l in examples["labels"]])
        return inputs, labels

    # This loop will evaluate the model for each event you want to test it on
    for test_event in args.test_on:
        # Some logging, will be useful to keep track of the different evaluation results
        if args.use_wandb:
            import wandb
            wandb.init(project="disaster-classification")
            wandb.config.train    = args.train_on
            wandb.config.eval     = test_event
            wandb.config.model    = args.model_name_or_path
            wandb.config.run      = "evaluation"

        # Load data to test on
        dataset = HumanitarianDataset(args.test_filename, disaster_names=test_event)
        testloader = DataLoader(dataset, batch_size=256, shuffle=False)

        # Run testing
        acc, f1 = validation(args, model, testloader, process_inputs)
        print(f"Results for the model trained on {', '.join(args.train_on)} evaluated on {test_event}:")
        print(f"\tAccuracy: {acc:.4f}")
        print(f"\tF1      : {f1:.4f}")

        if args.use_wandb:
            wandb.run.summary["best_accuracy"] = acc
            wandb.run.summary["best_f1"] = f1
