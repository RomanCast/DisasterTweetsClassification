import os


def get_model_name(args, saving=True, adversarial=False):
    """
    Create a name for your finetuned model / checkpoint. If saving is False, then this function will check that such a model exists, and raise an error if it does not exist.
    """
    domains = '_'.join(args.train_on) if args.train_on is not None else 'all'
    sources = '_' + '_'.join(args.train_sources) if args.train_sources is not None else ''
    model_name = f"{args.model_name_or_path}_{domains}{sources}"
    if adversarial:
        target_domains = '_'.join(args.target_events)  if args.target_events is not None else ''
        target_sources = '_'.join(args.target_sources) if args.target_events is not None else ''
        model_name += f"_target_{target_domains}{target_sources}"
    model_name += ".pt"
    if saving:
        return model_name
    else:
        if not os.path.exists(os.path.join(args.save_path, model_name)):
            raise FileNotFoundError(f"No checkpoint found for the name '{model_name}'.")
        return model_name
