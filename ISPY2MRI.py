# DATASET
import torch
import torchvision
from torchvision import transforms
from typing import Callable, Dict, Optional, Tuple, Union
from timm.models.layers import to_2tuple


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from PIL import Image
import time
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from utils import cached_load_png


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


# def split_data(
#     test_size,
#     data_path="/home/t-9bchoy/breast-cancer-treatment-prediction/processed_dataset.csv",
# ):
#     data = pd.read_csv(data_path)

#     if test_size > 0:
#         fit_data, test_data = stratified_group_split(
#             data, "PATIENT ID", "pcr", test_size
#         )
#     else:
#         fit_data = data
#         test_data = pd.DataFrame(columns=data.columns)

#     return fit_data, test_data


class ISPY2MRIRandomPatchSSLDataset(Dataset):
    def __init__(
        self,
        sequences,
        dataset="training",
        transform=None,
        image_size=256,
        patch_size: int = 64,
    ):
        if not isinstance(sequences, list):
            sequences = [sequences]
        substring = "|".join(sequences)
        if dataset == "training":
            data = pd.read_csv(
                "/home/t-9bchoy/breast-cancer-treatment-prediction/train_processed_dataset_T012_one_hot.csv"
            )
        elif dataset == "testing":
            data = pd.read_csv(
                "/home/t-9bchoy/breast-cancer-treatment-prediction/test_processed_dataset_T012_one_hot.csv"
            )
        self.xy = data[data["SHORTEN SEQUENCE"].str.contains(substring)]
        self.image_size = to_2tuple(image_size)
        # self.resize_and_random_crop = transforms.Compose(
        #     [transforms.Resize(self.image_size), transforms.RandomCrop(patch_size)]
        # )
        self.random_crop = transforms.RandomCrop(patch_size)
        self.transform = transform
        self.patch_size = patch_size
        w, h = self.image_size
        # possibly change to manually set num_patches
        self.num_patches = (w // patch_size) * (h // patch_size) * len(self.xy)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, index):
        row = self.xy.iloc[index // len(self.xy)]
        path = row["SEQUENCE PATH"]
        image = self.random_crop(self.load_png(path))
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


class ISPY2MRIDataSet(Dataset):
    def __init__(
        self,
        sequences,
        transform=None,
        data=None,
        dataset=None,
    ):
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

    def __getitem__(self, index):
        row = self.xy.iloc[index]
        path = row["SEQUENCE PATH"]
        image = cached_load_png(path)

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


def get_datasets(train_transform, sequences, val_transform, n_splits, random_state):

    fit_data = pd.read_csv(
        "/home/t-9bchoy/breast-cancer-treatment-prediction/train_processed_dataset_T012_one_hot.csv"
    )

    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fit_indices = np.arange(len(fit_data))
    # create validation and training splits for each fold
    # the data is all from the training data, but val splits use val transforms
    # the stratified group k fold class makes splits where patients do not cross splits
    # (based on patient id) and we aim for similar ratios of pcr/non-pcr
    fit_datasets = [
        (
            ISPY2MRIDataSet(
                sequences,
                transform=train_transform,
                data=fit_data.iloc[train_indices],
            ),
            ISPY2MRIDataSet(
                sequences,
                transform=val_transform,
                data=fit_data.iloc[val_indices],
            ),
        )
        for train_indices, val_indices in cv.split(
            fit_indices, fit_data["pcr"], fit_data["PATIENT ID"]
        )
    ]

    test_data = pd.read_csv(
        "/home/t-9bchoy/breast-cancer-treatment-prediction/test_processed_dataset_T012_one_hot.csv"
    )
    test_dataset = ISPY2MRIDataSet(
        sequences,
        data=test_data,
        transform=val_transform,
    )

    # log_summary("train + validation", fit_metadata)
    # log_summary("testing", test_metadata)

    return fit_datasets, test_dataset
