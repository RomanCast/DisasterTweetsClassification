import os


def get_model_name(args, saving=True):
    """
    Create a name for your finetuned model / checkpoint. If saving is False, then this function will check that such a model exists, and raise an error if it does not exist.
    """
    domains = '_'.join(args.train_on) if args.train_on is not None else 'all'
    model_name = f"{args.model_name_or_path}_{domains}.pt"
    if saving:
        return model_name
    else:
        if not os.path.exists(os.path.join(args.save_path, model_name)):
            raise FileNotFoundError(f"No checkpoint found for the name '{model_name}'.")
        return model_name
