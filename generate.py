import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import json

from model import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='checkpoint/model_final.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=9, help='Number of generated outputs')
parser.add_argument('-eye_color', default='blue', help='Eye color')
parser.add_argument('-hair_color', default='blonde', help='Hair color')
parser.add_argument('-load_json', default='data/animegan_params.json', help='Load path for params json.')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Load color2ind dictionary from json parameter file.
with open(args.load_json, 'r') as info_file:
            info = json.load(info_file)
            color2ind = info['color2ind']

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

print(args.num_output)
# Get latent vector Z from unit normal distribution.
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

# To create onehot embeddings for the condition labels.
onehot = torch.zeros(params['vocab_size'], params['vocab_size'])
onehot = onehot.scatter_(1, torch.LongTensor([i for i in range(params['vocab_size'])]).view(params['vocab_size'], 1), 1).view(params['vocab_size'], params['vocab_size'], 1, 1)

# Create input conditions vectors.
input_condition = torch.cat((torch.ones(int(args.num_output), 1)*color2ind[args.eye_color], 
                            torch.ones(int(args.num_output), 1)*color2ind[args.hair_color]),
                            dim=1).type(torch.LongTensor)

# Generate the onehot embeddings for the conditions.
eye_ohe = onehot[input_condition[:, 0]].to(device)
hair_ohe = onehot[input_condition[:, 1]].to(device)

# Turn off gradient calculation to speed up the process.
with torch.no_grad():
    # Get generated image from the noise vector using
    # the trained generator.
    generated_img = netG(noise, eye_ohe, hair_ohe).detach().cpu()

# Display the generated image.
plt.axis("off")
#plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, nrow=int(np.sqrt(int(args.num_output))), padding=2, normalize=True), (1,2,0)))

plt.show()