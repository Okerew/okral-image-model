import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
from PIL import Image
import spacy
import time


timestamp = time.time()
current_date = time.ctime(timestamp)
# Define the GAN architecture
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Custom dataset class
class TextImageDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.nlp = spacy.load('en_core_web_sm')
        self.data = self.load_data()

    def load_data(self):
        data = []
        for file_name in os.listdir(self.data_folder):
            with open(os.path.join(self.data_folder, file_name), 'r') as f:
                file_contents = json.load(f)
                for entry in file_contents:
                    data.append((entry['text'], entry['image']))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, image_path = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return text, image


# Preprocessing and loading data
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_folder = 'training_data'
dataset = TextImageDataset(data_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize models
nz = 100  # Size of z latent vector
ngf = 64
ndf = 64
nc = 3  # Number of color channels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(nc, ndf).to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Create directories if they don't exist
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('model'):
    os.makedirs('model')

# Training loop
num_epochs = 100
sample_images = []

for epoch in range(num_epochs):
    for i, (texts, images) in enumerate(dataloader):
        # Update Discriminator
        netD.zero_grad()
        real_images = images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size,), 1, dtype=torch.float, device=device)
        output = netD(real_images).view(-1)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        labels.fill_(0)
        output = netD(fake_images.detach()).view(-1)
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update Generator
        netG.zero_grad()
        labels.fill_(1)
        output = netD(fake_images).view(-1)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")
    if epoch % 10 == 0:
        vutils.save_image(fake_images, f'output/samples_epoch_{epoch}_{current_date}.png',
                          normalize=True)
        sample_images.append(fake_images[0].cpu())

# Save the models
torch.save(netG.state_dict(), 'model/generator.pt')
torch.save(netD.state_dict(), 'model/discriminator.pt')

# Stack 10 sample images along the channel dimension
final_image = torch.cat(sample_images[:10], dim=0)  # Stack images along the channel dimension

# Save the final image tensor
torch.save(final_image, f'output/final_image{current_date}.pt')