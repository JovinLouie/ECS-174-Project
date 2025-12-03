import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
        #Encoder
        #in 400x400x3, works for inputs divisible by 16
        self.e1a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e4a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e5a = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e5b = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        
        #Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d1a = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d1b = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d2a = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d2b = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d3a = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d4a = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d4b = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        #Output Layer
        self.out = nn.Conv2d(64, n_class, kernel_size=1)

        # Better initialization for binary segmentation
        # Bias the output towards negative (predicting background by default)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, -2.0)  # Start biased toward 0 (background)
    
    def forward(self, x):
        #Encoder
        xe1a = torch.relu(self.e1a(x))
        xe1b = torch.relu(self.e1b(xe1a))
        xp1 = self.pool1(xe1b)

        xe2a = torch.relu(self.e2a(xp1))
        xe2b = torch.relu(self.e2b(xe2a))
        xp2 = self.pool2(xe2b)

        xe3a = torch.relu(self.e3a(xp2))
        xe3b = torch.relu(self.e3b(xe3a))
        xp3 = self.pool3(xe3b)

        xe4a = torch.relu(self.e4a(xp3))
        xe4b = torch.relu(self.e4b(xe4a))
        xp4 = self.pool4(xe4b)

        xe5a = torch.relu(self.e5a(xp4))
        xe5b = torch.relu(self.e5b(xe5a))

        #Decoder
        xu1 = self.up1(xe5b)
        xu11 = torch.cat([xu1, xe4b], dim=1)
        xd1a = torch.relu(self.d1a(xu11))
        xd1b = torch.relu(self.d1b(xd1a))

        xu2 = self.up2(xd1b)
        xu21 = torch.cat([xu2, xe3b], dim=1)
        xd2a = torch.relu(self.d2a(xu21))
        xd2b = torch.relu(self.d2b(xd2a))

        xu3 = self.up3(xd2b)
        xu31 = torch.cat([xu3, xe2b], dim=1)
        xd3a = torch.relu(self.d3a(xu31))
        xd3b = torch.relu(self.d3b(xd3a))

        xu4 = self.up4(xd3b)
        xu41 = torch.cat([xu4, xe1b], dim=1)
        xd4a = torch.relu(self.d4a(xu41))
        xd4b = torch.relu(self.d4b(xd4a))

        #Output Layer
        xout = self.out(xd4b)

        return xout