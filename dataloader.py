import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class AnimeDataset(Dataset):
	"""Anime Dataset"""

	def __init__(self, csv_file, root_dir, transform=None):
		self.color_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.color_frame)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir,
		                        str(self.color_frame.iloc[idx, 0])+".jpg")

		image = io.imread(img_name)
		colors = self.color_frame.iloc[idx, 1:].values

		sample = {'image': image, 'colors': colors}

		if self.transform:
			sample = self.transform(sample)

		return sample

"""
anime_dataset = AnimeDataset(csv_file='data/clean_labels.csv',
	                         root_dir='data/faces/')

fig = plt.figure()

for i in range(len(anime_dataset)):
	sample = anime_dataset[i]

	print(i, sample['image'].shape, sample['colors'])

	ax = plt.subplot(1, 4, i+1)
	plt.tight_layout()
	plt.imshow(sample['image'])
	ax.set_title('Sample #{}'.format(i))
	ax.axis('off')

	if i ==3:
		plt.show()
		break
"""