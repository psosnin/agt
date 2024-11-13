"""To use this dataset, please first extract Ch2_002_out.zip to the directory datasets/Ch2_002_out"""

import math
import PIL
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from itertools import accumulate
from torchvision.transforms.functional import pil_to_tensor
import pandas as pd


def getSelfDrive(
    train_batchsize,
    test_batchsize=500,
    data_path="datasets/Ch2_002_out",
    normalise=True,
    video_index=[1, 2, 3],
    input_size=(200, 66),
    **kwargs,
):
    """
    Return dataloaders for the self-driving steering angle prediction dataset.
    """
    dataset = SelfDriveDataset(data_path, video_index, normalise, input_size)

    split = [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
    train_data, test_data = torch.utils.data.random_split(dataset, split)

    train_loader = DataLoader(train_data, batch_size=train_batchsize, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batchsize, shuffle=True)

    return train_loader, test_loader


class SelfDriveDataset(torch.utils.data.Dataset):
    """
    Dataset class for the self-driving steering angle prediction dataset.
    """

    def __init__(self, data_path, video_index=[1, 2, 3], normalise=False, input_size=(200, 66)):
        """
        Parameters:
            data_path: path to the directory containing the data
            video_index: list of videos to include (from 1-6)
        """
        super().__init__()
        self.video_index = video_index
        self.data_path = data_path
        self.input_size = input_size
        self.normalise = normalise
        self.timestamps = []
        self.angles = []
        self.lengths = []

        # read the timestamps and steering angles from the csv
        for index in video_index:
            df = pd.read_csv(f"{data_path}/HMB_{index}/interpolated.csv")
            self.timestamps.append(df[df["frame_id"] == "center_camera"]["timestamp"].values)
            self.angles.append(torch.Tensor(df[df["frame_id"] == "center_camera"]["angle"].values).float())
            self.lengths.append(len(self.timestamps[-1]))

        self.lengths = list(accumulate(self.lengths))

    def __len__(self):
        return self.lengths[-1]

    def __getitem__(self, idx):
        # work out which video the index belongs to
        i = 0
        while self.lengths[i] < idx:
            i += 1
            if i == len(self.lengths):
                raise IndexError

        timestamp = self.timestamps[i][idx - self.lengths[i]]
        angle = self.angles[i][idx - self.lengths[i]]

        image_pil = PIL.Image.open(f"{self.data_path}/HMB_{self.video_index[i]}/center/{timestamp:d}.jpg")
        image_pil = image_pil.resize(
            size=self.input_size,
            resample=PIL.Image.BILINEAR,
            box=(0, 80, 640, 480),
        )
        image = pil_to_tensor(image_pil) / 255
        if self.normalise:
            image = (image - image.mean()) / image.std()
        return image.transpose(-2, -1), angle


def plotImage(image, y=None):
    """
    Plot an image from a dataset. y is an optional steering angle for the self-drive dataset.
    """
    image_pil = transforms.functional.to_pil_image(image.transpose(-2, -1))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(image_pil)
    if y is not None:
        ax.plot([100, 100 - math.tan(y.item() * torch.pi) * image.shape[-1]], [image.shape[-1], 0], lw=2, color="g")
    ax.set_axis_off()
    ax.set_ylim(image.shape[-1], 0)
    ax.set_xlim(image.shape[-2], 0)
    plt.show()
