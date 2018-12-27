import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(w):
    classname = w.__class__.__name__
    if (type(w) == nn.ConvTranspose2d or type(w) == nn.Conv2d):
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif (type(w) == nn.BatchNorm2d):
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)
    elif (type(w) == nn.Linear):
        nn.init.normal_(w.weight.data, 0.0, 0.02)

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.fc_embed1 = nn.Linear(params['embedding_size'], 128, bias=False)
        self.fc_embed2 = nn.Linear(params['embedding_size'], 128, bias=False)

        # Input is the latent vector Z + Conditions.
        self.tconv1 = nn.ConvTranspose2d(params['nz'] + 128*2, params['ngf']*8, 
                                           kernel_size=4, stride=1, 
                                           padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['nc'],
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x, y1, y2):
        
        y1 = F.leaky_relu(self.fc_embed1(y1.squeeze()), 0.2, True)
        y2 = F.leaky_relu(self.fc_embed2(y2.squeeze()), 0.2, True)

        x = torch.cat((x, y1.view(-1, 128, 1, 1), y2.view(-1, 128, 1, 1)), dim=1)

        x = F.leaky_relu(self.bn1(self.tconv1(x)), 0.2, True)

        x = F.leaky_relu(self.bn2(self.tconv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.tconv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.tconv4(x)), 0.2, True)

        x = F.tanh(self.tconv5(x))

        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params['ndf']*8 + 128*2, params['ndf']*8, 1, 1, 0, bias=False)
        self.bn5 = self.bn5 = nn.BatchNorm2d(params['ndf']*8)

        self.conv6 = nn.Conv2d(params['ndf']*8, 1, 4, 1, 0, bias=False)

        self.fc_embed1 = nn.Linear(params['embedding_size'], 128, bias=False)
        self.fc_embed2 = nn.Linear(params['embedding_size'], 128, bias=False)

    def forward(self, x, y1, y2):
        img = x

        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        y1 = F.leaky_relu(self.fc_embed1(y1.squeeze()), 0.2, True)
        y2 = F.leaky_relu(self.fc_embed2(y2.squeeze()), 0.2, True)
        y = torch.cat((y1, y2), dim=1)
        y = y.view(y.size(0), y.size(1), 1, 1)

        y_fill = y.repeat(1, 1, 4, 4)
        x = torch.cat((x, y_fill), dim=1)

        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)
        x = F.sigmoid(self.conv6(x))

        return x
