import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from six import iteritems

class AnimeDataset(Dataset):
	"""Anime Dataset"""

	def __init__(self, csv_file, root_dir, param_file, transform=None):
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
		img_name = os.path.join(self.root_dir,
		                        str(self.color_frame.iloc[idx, 0])+".jpg")

		image = io.imread(img_name)
		colors = self.color_frame.iloc[idx, 1:].values
		colors = [self.color2ind[color] for color in colors]


		sample = {'image': image, 'colors': colors}

		if self.transform:
			sample = self.transform(sample)

		return sample

"""
anime_dataset = AnimeDataset(csv_file='data/clean_labels.csv',
	                         root_dir='data/faces/',
	                         param_file='data/animegan_params.json')

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
		print(sample)
		plt.show()
		break
"""