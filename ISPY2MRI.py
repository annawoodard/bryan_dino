# DATASET
import torch
import torchvision
from torchvision import transforms
from typing import Callable, Dict, Optional, Tuple, Union
from timm.models.layers import to_2tuple


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
from PIL import Image
import time
from sklearn.model_selection import train_test_split


def stratified_group_split(
    # adapted from https://stackoverflow.com/a/64663756/5111510
    samples: pd.DataFrame,
    group: str,
    stratify_by: str,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = samples[group].drop_duplicates()
    stratify = samples.drop_duplicates(group)[stratify_by].to_numpy()
    groups_train, groups_test = train_test_split(
        groups, stratify=stratify, test_size=test_size
    )

    samples_train = samples.loc[lambda d: d[group].isin(groups_train)]
    samples_test = samples.loc[lambda d: d[group].isin(groups_test)]

    return samples_train, samples_test


def split_data(
    test_size,
    data_path="/home/t-9bchoy/breast-cancer-treatment-prediction/processed_dataset.csv",
):
    data = pd.read_csv(data_path)

    if test_size > 0:
        fit_data, test_data = stratified_group_split(
            data, "PATIENT ID", "pcr", test_size
        )
    else:
        fit_data = data
        test_data = pd.DataFrame(columns=data.columns)

    return fit_data, test_data


class ISPY2MRIDataSet(Dataset):
    def __init__(self, sequences, dataset="training", transform=None, image_size=448):
        if not isinstance(sequences, list):
            sequences = [sequences]
        substring = "|".join(sequences)
        # data = pd.read_csv(
        #     "/home/t-9bchoy/breast-cancer-treatment-prediction/processed_dataset.csv"
        # )
        # using the combined dataset
        if dataset == "training":
            data = pd.read_csv(
                "/home/t-9bchoy/breast-cancer-treatment-prediction/train_processed_dataset_T012_one_hot.csv"
            )
        elif dataset == "testing":
            data = pd.read_csv(
                "/home/t-9bchoy/breast-cancer-treatment-prediction/test_processed_dataset_T012_one_hot.csv"
            )
        self.xy = data[data["SHORTEN SEQUENCE"].str.contains(substring)]

        self.n_samples = len(self.xy)
        self.transform = transform
        self.resize = transforms.Resize(to_2tuple(image_size))

    def __getitem__(self, index):
        row = self.xy.iloc[index]
        path = row["SEQUENCE PATH"]
        image = self.resize(self.load_png(path))
        if self.transform == None:
            return image, row["pcr"]
        else:
            return self.transform(image), row["pcr"]

    def load_png(self, filename):
        try:
            image = Image.open(filename).convert("L")
        except OSError:
            time.sleep(2)
            image = Image.open(filename).convert("L")
        return image

    def __len__(self):
        return self.n_samples
