import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from six import iteritems

class AnimeDataset(Dataset):
    """Anime Dataset"""

    def __init__(self, csv_file, root_dir, param_file, transform):
        self.color_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.param_file = param_file

        with open(self.param_file, 'r') as info_file:
            info = json.load(info_file)

            for key, value in iteritems(info):
                setattr(self, key, value)

        color_count = len(self.color2ind)
        self.color2ind['<S>'] = color_count + 1
        self.color2ind['</S>'] = color_count + 2
        self.start_token = self.color2ind['<S>']
        self.end_token = self.color2ind['</S>']

        self.ind2color = {
            int(ind): color
            for color, ind in iteritems(self.color2ind)
        }

    def __len__(self):
        return len(self.color_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                str(self.color_frame.iloc[idx, 0])+".jpg")

        image = Image.open(img_path)
        colors = self.color_frame.iloc[idx, 1:].values
        colors = [self.color2ind.get(color, self.color2ind['<UNK>']) for color in colors]
        colors = torch.LongTensor(colors)

        transformed_images = self.transform(image)

        sample = {'image': transformed_images, 'colors': colors}

        return sample 


"""
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))])

anime_dataset = AnimeDataset(csv_file='data/clean_labels.csv',
                             root_dir='data/faces/',
                             param_file='data/animegan_params.json',
                             transform=transform)

dataloader = DataLoader(anime_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['colors'].shape)

    # observe 4th batch and stop.
    if i_batch == 3:
        print(sample_batched['colors'])
        plt.figure()
        grid = utils.make_grid(sample_batched['image'])
        plt.imshow(grid.numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.ioff()
        plt.show()

        break
"""