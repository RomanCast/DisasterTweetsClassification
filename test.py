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

def test(args, model, testloader):
    """
    Uses the model and a validation set to assess the performance of the model through training.

    inputs:
        args: Namespace object containing necessary variables.
        model: a Pytorch model that is being trained
        testloader: a DataLoader (see PyTorch docs) containing the validation / develoment set

    outputs:
        accuracy and f1 score of the model on the validation data
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, sample in enumerate(testloader):
            inputs, labels = process_inputs(sample)
            inputs = {key: val.to(args.device) for key, val in inputs.items()}
            labels = labels.to(args.device)

            outputs = model(**inputs, labels=labels)

            preds = list(outputs.logits.cpu().max(dim=1)[1].numpy())
            all_preds += preds
            all_labels += list(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = (all_preds == all_labels).sum() / all_preds.shape[0]
    return acc, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--train_filename", type=str, nargs='+', help="The dataset on which the model was trained on. Used to retrieve the labels")
    parser.add_argument("--test_filename", type=str)
    parser.add_argument("--train_on", type=str, nargs="+", help="List of disaster on which the model was trained.")
    parser.add_argument("--test_on", type=str, help="Disaster on which to test the model.")
    parser.add_argument("--use_wandb", action='store_true')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.save_path = "models"

    # Some logging, will be useful to keep track of the different evaluation results
    if args.use_wandb:
        import wandb
        wandb.init(project="disaster-classification")
        wandb.config.train    = args.train_on
        wandb.config.eval     = args.test_on
        wandb.config.model    = args.model_name_or_path
        wandb.config.run      = "evaluation"

    # Get the model name on the existing folder containing models. If it does not exist, this will throw an exception
    args.finetuned_checkpoint = get_model_name(args, saving=False)

    # Get the number of unique labels
    unique_labels = json.load(open("data/humanitarian_labels.json", 'r'))
    num_labels = len(unique_labels)
    label_to_id = {l: i for i, l in enumerate(unique_labels)}

    # Load data to test on
    dataset = HumanitarianDataset(args.test_filename, disaster_names=args.test_on)
    testloader = DataLoader(dataset, batch_size=256, shuffle=False)

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

    # Run testing
    acc, f1 = test(args, model, testloader)
    print(f"Results for the model trained on {', '.join(args.train_on)} evaluated on {args.test_on}:")
    print(f"\tAccuracy: {acc:.4f}")
    print(f"\tF1      : {f1:.4f}")

    if args.use_wandb:
        wandb.run.summary["accuracy"] = acc
        wandb.run.summary["f1"] = f1
