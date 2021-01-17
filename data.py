from torch.utils.data import DataLoader, Dataset
import pandas as pd


class HumanitarianDataset(Dataset):
    def __init__(self, filename, disaster_names=None, source_names=None, preprocess=True):
        if type(disaster_names) == str:
            disaster_names = [disaster_names]
        if type(source_names) == str:
            sources_names = [source_names]
        # Load several datasets and concatenate them
        if type(filename) == list:
            all_data_list = []
            for fn in filename:
                all_data_list += [self._load_data(fn, disaster_names, source_names, preprocess)]
            all_data = pd.concat(all_data_list, axis=0, ignore_index=True)
        # Load a single dataset
        else:
            all_data = self._load_data(filename, disaster_names, source_names, preprocess)
        self.data = all_data["text"].tolist()
        self.labels = all_data["class_label"].tolist()

    def __getitem__(self, idx):
        return {"data": self.data[idx], "labels": self.labels[idx]}

    def __len__(self):
        return len(self.data)

    def _load_data(self, filename, names, sources, preprocess):
        """
        inputs:
            filename: a string containing the dataset path
            names: a list of strings with the events to consider
            preprocess: a boolean indicating if we process the text data or not

        outputs:
            a dataframe with our data with only some rows.
        """
        all_data = pd.read_csv(filename, sep="\t")
        if sources is not None:
            # Select rows from specific sources
            all_data = all_data[all_data["source"].isin(sources)]
        if names is not None:
            # Select rows depending on the type of disaster
            all_data = all_data[all_data["event"].isin(names)]
        if preprocess:
            # Some simple preprocessing, removing the "#", "RT", "@...", "http://t.co/etc..", non ascii characters, and replace multiple spaces by one
            all_data["text"] = all_data["text"].str.replace(r"RT|#|@\w+:?|https?\S+", "") \
                                               .str.replace(r"[^\x00-\x7F]+", "") \
                                               .str.replace("\s+", " ") \
                                               .str.strip()

        return all_data


if __name__ == "__main__":
    # Print the text from the "light" dataset (used for debugging). This is just to ensure that text processing worked correctly
    dataset = HumanitarianDataset("data/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_train_light.tsv")
    dataloader = DataLoader(dataset, batch_size=1)

    for sample in dataloader:
        print(sample["data"][0])
        print()
