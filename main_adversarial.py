import os
import json
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import time

from data import HumanitarianDataset
from utils import get_model_name
from model import DomainAdversarialRoberta

def train(args, model, optimizer,
          trainloader_src, trainloader_tgt,
          validloader, process_inputs):
    """
    Train the model on the training data for a number of epochs specified in the arguments.

    inputs:
        args: Namespace object containing necessary variables.
        model: a PyTorch model being trained. Preferably a HuggingFace model since it needs to output an object with attributes "loss" and "logits"
        optimizer: the optimizer used to train the model
        trainloader_src: a DataLoader object containing the training data in the source domain
        trainloader_tgt: a DataLoader object containing the training data in the target domain
        validloader: a DataLoader object containing the validation / development data
        process_inputs: a function that processes each batch
    """
    model.train()
    for epoch in range(args.n_epochs):
        running_loss = 0.
        best_f1 = 0.
        for i, (sample_src, sample_tgt) in enumerate(zip(trainloader_src, trainloader_tgt)):
            # Process the inputs from words to tensors
            inputs_src, labels = process_inputs(sample_src)
            inputs_tgt, _      = process_inputs(sample_tgt)
            # Load them and the labels on GPUs if available
            inputs_src = {key         : val.to(args.device) for key, val in inputs_src.items()}
            inputs_tgt = {key + '_tgt': val.to(args.device) for key, val in inputs_tgt.items()}
            labels = labels.to(args.device)
            # We put label 1 to source domain, 0 to target domain
            domains = torch.cat([torch.ones(inputs_src['input_ids'].size(0)),
                                 torch.zeros(inputs_tgt['input_ids_tgt'].size(0))], 0).long()
            domains = domains.to(args.device)
            optimizer.zero_grad()
            # Run the inputs through the model: that will compute logits necessary for inference, as well as the loss
            outputs = model(**inputs_src, **inputs_tgt, labels=labels, domains=domains, alpha=args.alpha)
            # Perform backpropagation to update your model weights
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Every args.val_metrics, evaluate your model
            if (i + 1) % args.val_metrics == 0:
                acc, f1 = validation(args, model, validloader, process_inputs)
                print(f"[{epoch}, {i:5d}] : F1 = {f1:.4f}; Acc = {acc:.4f}; Tr loss = {running_loss / args.val_metrics:.4f}")
                if args.use_wandb:
                    batch = min(len(trainloader_src), len(trainloader_tgt)) * epoch + i
                    wandb.log({'accuracy': acc, 'f1': f1, 'training_loss': running_loss, 'batch': batch})
                running_loss = 0.
                model.train()

        acc, f1 = validation(args, model, validloader, process_inputs)
        # We only save the best model given by the validation data
        if f1 > best_f1:
            best_f1 = f1
            if args.use_wandb:
                wandb.run.summary["best_f1"] = f1
                wandb.run.summary["best_accuracy"] = acc
                wandb.run.summary["best_epoch"] = epoch
            # Save the model
            torch.save(model.state_dict(), os.path.join(args.save_path, get_model_name(args, adversarial=True)))


def validation(args, model, validloader, process_inputs):
    """
    Uses the model and a validation set to assess the performance of the model through training.

    inputs:
        args: Namespace object containing necessary variables.
        model: a Pytorch model that is being trained
        validloader: a DataLoader (see PyTorch docs) containing the validation / develoment set

    outputs:
        accuracy and f1 score of the model on the validation data
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, sample in enumerate(validloader):
            inputs, labels = process_inputs(sample)
            inputs = {key: val.to(args.device) for key, val in inputs.items()}
            empty_inputs = {key + '_tgt': None for key, val in inputs.items()}
            labels = labels.to(args.device)
            domains = None

            outputs = model(**inputs, **empty_inputs, labels=labels, domains=domains, alpha=args.alpha)

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
    parser.add_argument("--dataset_filename", type=str, nargs="+", default=["data/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_train_light.tsv",
                                                                            "data/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_dev_light.tsv"])
    parser.add_argument("--train_on", default=None, nargs="+", type=str, help="List of names of disasters on which to train the model")
    parser.add_argument("--train_sources", default=None, nargs="+", type=str, help="List of database sources on which to train the model")
    parser.add_argument("--target_events", default=None, nargs="+", type=str, help="List of target event names")
    parser.add_argument("--target_sources", default=None, nargs="+", type=str, help="List of target database sources")
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save_path", type=str, default="models")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--use_wandb", action='store_true')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.val_metrics = 300
    args.train_prop = 0.8
    args.alpha = 1e-3
    # In the case of domain adaptation, we fixed what model we're using since it is not an "AutoModel"
    args.model_name_or_path = "roberta-base"

    # Some logging, will be useful to keep track of the different results
    if args.use_wandb:
        import wandb
        wandb.init(project="disaster-classification")
        wandb.config.train       = args.train_on
        wandb.config.sources     = args.train_sources
        wandb.config.tgt_events  = args.target_events
        wandb.config.tgt_sources = args.target_sources
        wandb.config.lr          = args.lr
        wandb.config.bs          = args.batch_size
        wandb.config.model       = args.model_name_or_path
        wandb.config.n_epochs    = args.n_epochs
        wandb.config.seed        = args.seed
        wandb.config.run         = "training"

    # For reproducibility purpose
    set_seed(args.seed)

    # Load the data in DataLoaders
    dataset     = HumanitarianDataset(args.dataset_filename, event_names=args.train_on, source_names=args.train_sources)
    dataset_tgt = HumanitarianDataset(args.dataset_filename, event_names=args.target_events, source_names=args.target_sources)

    # Split the dataset into training and testing splits
    train_length = int(len(dataset) * args.train_prop)
    train_dataset, val_dataset = random_split(dataset, lengths=[train_length, len(dataset) - train_length])
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Also create a dataloader for the data in the target domain
    trainloader_tgt = DataLoader(dataset_tgt, batch_size=args.batch_size, shuffle=True)

    # We want to use the validation data at least twice per epoch
    args.val_metrics = min(args.val_metrics, len(trainloader) // 2)

    # Get the number of unique labels to initialize the model
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
    model = DomainAdversarialRoberta(config).to(args.device)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # We process each batch with this function. This way, we can pad to the specific length of every batch and avoid unnecessary long inputs
    def process_inputs(examples):
        inputs = tokenizer(examples["data"], padding=True, max_length=None, truncation=True)
        inputs = {key: torch.tensor(val) for key, val in inputs.items()}
        labels = torch.tensor([label_to_id[l] for l in examples["labels"]])
        return inputs, labels

    train(args, model, optimizer, trainloader, trainloader_tgt, validloader, process_inputs)
