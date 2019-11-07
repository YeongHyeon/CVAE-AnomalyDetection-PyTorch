import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(object):

    def __init__(self, height, width, channel, device, ngpu, ksize, z_dim, learning_rate=1e-3):

        self.height, self.width, self.channel = height, width, channel
        self.device, self.ngpu = device, ngpu
        self.ksize, self.z_dim, self.learning_rate = ksize, z_dim, learning_rate

        self.encoder = \
            Encoder(height=self.height, width=self.width, channel=self.channel, \
            ngpu=self.ngpu, ksize=self.ksize, z_dim=self.z_dim).to(self.device)
        self.decoder = \
            Decoder(height=self.height, width=self.width, channel=self.channel, \
            ngpu=self.ngpu, ksize=self.ksize, z_dim=self.z_dim).to(self.device)
        if(self.device.type == 'cuda') and (self.model.ngpu > 0):
            self.encoder = nn.DataParallel(self.encoder, list(range(self.encoder.ngpu)))
            self.decoder = nn.DataParallel(self.decoder, list(range(self.decoder.ngpu)))

        num_params = 0
        for model in [self.encoder, self.decoder]:
            for p in model.parameters():
                num_params += p.numel()
            print(model)
        print("The number of parameters: %d" %(num_params))

        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.SGD(self.params, lr=1e-5)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Encoder(nn.Module):

    def __init__(self, height, width, channel, ngpu, ksize, z_dim):
        super(Encoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu, self.ksize, self.z_dim = ngpu, ksize, z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.CELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.CELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.CELU(),
            Flatten(),
            nn.Linear((self.height//(2**2))*(self.width//(2**2))*self.channel*64, 512),
            nn.CELU(),
            nn.Linear(512, self.z_dim*2),
        )

    def split_z(self, z):

        z_mu = z[:, :self.z_dim]
        z_sigma = torch.clamp(z[:, self.z_dim:], min=1e-12, max=1-(1e-12))

        return z_mu, z_sigma

    def sample_z(self, mu, sigma):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)

        return mu + (eps * std)

    def forward(self, input):

        z_params = self.encoder(input)
        z_mu, z_sigma = self.split_z(z=z_params)
        z_enc = self.sample_z(mu=z_mu, sigma=z_sigma)

        return z_enc, z_mu, z_sigma

class Decoder(nn.Module):

    def __init__(self, height, width, channel, ngpu, ksize, z_dim):
        super(Decoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu, self.ksize, self.z_dim = ngpu, ksize, z_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim*2, 512),
            nn.Linear(512, (self.height//(2**2))*(self.width//(2**2))*self.channel*64),
            nn.CELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.CELU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=self.ksize, stride=2, padding=self.ksize//2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.CELU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=self.ksize, stride=2, padding=self.ksize//2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.ksize, stride=2, padding=self.ksize//2),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.Sigmoid(),
        )

    def forward(self, input):

        x_hat = self.decoder(input=input)

        return x_hat
