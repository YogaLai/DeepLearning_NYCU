import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, img_shape, ndf):
        super(Discriminator, self).__init__()
        self.H, self.W, self.C = img_shape
        self.condtion_embedd = nn.Sequential(
            nn.Linear(24, 1 * self.H * self.W),
            nn.LeakyReLU()
        )
        self.main = nn.Sequential(
            # input is (img_channel + 1) x 64 x 64 => 1 is condition
            nn.Conv2d(self.C*2, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, condition):
        condition = self.condtion_embedd(condition)
        condition = condition.view(-1, 1, self.H, self.W)
        input = torch.cat((input,condition), axis=1)
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()

        self.nc = nc
        self.condition_embedd = nn.Sequential(
            nn.Linear(24, nc),
            nn.LeakyReLU()
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz + nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ( rgb_channel: 3) x 64 x 64
        )

    def forward(self, input, condition):
        condition = self.condition_embedd(condition)
        condition = condition.view(-1, self.nc, 1, 1)
        input = torch.cat((input, condition), axis=1)
        return self.main(input)