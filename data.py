from torch.utils.data import DataLoader, Dataset
import pandas as pd


class HumanitarianDataset(Dataset):
    def __init__(self, filename, disaster_names=None, preprocess=True):
        if type(disaster_names) == str:
            disaster_names = [disaster_names]
        if type(filename) == list:
            all_data_list = []
            for fn in filename:
                all_data_list += [self._load_data(fn, disaster_names, preprocess)]
            all_data = pd.concat(all_data_list, axis=0)
        else:
            all_data = self._load_data(filename, disaster_names, preprocess)
        self.data = all_data["text"].tolist()
        self.labels = all_data["class_label"].tolist()

    def __getitem__(self, idx):
        return {"data": self.data[idx], "labels": self.labels[idx]}

    def __len__(self):
        return len(self.data)

    def get_unique_labels(self):
        """
        outputs:
            a list containing all the labels sorted by name
        """
        return sorted(list(pd.unique(self.labels)))

    def _load_data(self, filename, names, preprocess):
        """
        inputs:
            filename: a string containing the dataset path
            names: a list of strings with the events to consider
            preprocess: a boolean indicating if we process the text data or not

        outputs:
            a dataframe with our data with only some rows.
        """
        all_data = pd.read_csv(filename, sep="\t")
        if names is not None:
            # Select only some rows
            all_data = all_data[all_data["event"].isin(names)]
        if preprocess:
            # Some simple preprocessing, removing the "#", "RT", "@...", "http://t.co/etc.."
            all_data["text"] = all_data["text"].str.replace(r"RT|#|@\w+:?|https?\S+", "") \
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
