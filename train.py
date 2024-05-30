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
import math

timestamp = time.time()
current_date = time.ctime(timestamp)


class Generator(nn.Module):
    """
    Generator network for GAN.

    Args:
        nz (int): Size of the latent vector (input noise).
        ngf (int): Size of feature maps in generator.
        nc (int): Number of channels in the output image.
    """
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
        """
        Forward pass through the generator.

        Args:
            input (torch.Tensor): Input tensor (latent vector).

        Returns:
            torch.Tensor: Generated image.
        """
        return self.main(input)


class Discriminator(nn.Module):
    """
    Discriminator network for GAN.

    Args:
        nc (int): Number of channels in the input image.
        ndf (int): Size of feature maps in discriminator.
    """
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
        """
        Forward pass through the discriminator.

        Args:
            input (torch.Tensor): Input tensor (image).

        Returns:
            torch.Tensor: Discriminator output.
        """
        return self.main(input)


class TextImageDataset(Dataset):
    """
    Custom dataset for loading text and image pairs.

    Args:
        data_folder (str): Path to the folder containing the data.
        transform (callable, optional): Optional transform to be applied on an image sample.
    """

    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.nlp = spacy.load('en_core_web_sm')
        self.data = self.load_data()

    def load_data(self):
        """
        Load data from the data folder.

        Returns:
            list: List of tuples containing text, image path, and identifier.
        """
        data = []
        for file_name in os.listdir(self.data_folder):
            with open(os.path.join(self.data_folder, file_name), 'r') as f:
                file_contents = json.load(f)
                for idx, entry in enumerate(file_contents):
                    data.append((entry['text'], entry['image'], file_name + '_' + str(idx)))
        return data

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing processed text, image, and identifier.
        """
        text, image_path, identifier = self.data[idx]

        # Process the text using SpaCy
        doc = self.nlp(text)
        processed_text = [
            (token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
            for token in doc]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return processed_text, image, identifier


class ResidualBlock(nn.Module):
    """
    Residual block for the diffusion model.

    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying residual block.
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class DiffusionModel(nn.Module):
    """
    Diffusion model for image denoising.

    Args:
        nc (int): Number of channels in the input image.
        ngf (int, optional): Size of feature maps. Default is 64.
    """
    def __init__(self, nc, ngf=64):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            ResidualBlock(ngf * 2)
        )

        self.decoder = nn.Sequential(
            ResidualBlock(ngf * 2),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Forward pass through the diffusion model.

        Args:
            x (torch.Tensor): Input tensor (noisy image).

        Returns:
            torch.Tensor: Output tensor (denoised image).
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torch.clamp(decoded, 0, 1)
        return decoded


class CosineScheduler:
    """
    Cosine scheduler for controlling noise level during training.

    Args:
        num_steps (int): Number of steps in the noise schedule.
        max_noise (float, optional): Maximum noise level. Default is 1.0.
        min_noise (float, optional): Minimum noise level. Default is 0.0.
    """
    def __init__(self, num_steps, max_noise=1.0, min_noise=0.0):
        self.num_steps = num_steps
        self.max_noise = max_noise
        self.min_noise = min_noise

    def step(self, step):
        """
        Compute the noise level for a given step.

        Args:
            step (int): Current step in the schedule.

        Returns:
            float: Noise level for the current step.
        """
        alpha = self.min_noise + 0.5 * (self.max_noise - self.min_noise) * (1 + math.cos(step * math.pi / self.num_steps))
        return alpha


def add_noise(images, noise_level):
    """
    Add Gaussian noise to images.

    Args:
        images (torch.Tensor): Input images.
        noise_level (float): Standard deviation of the noise.

    Returns:
        torch.Tensor: Noisy images.
    """
    noise = torch.randn_like(images) * noise_level
    noisy_images = images + noise
    return noisy_images


# Preprocessing and loading data
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_folder = 'training_data'
dataset = TextImageDataset(data_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for texts, images, identifier in dataloader:
    print(f"Processed Text: {texts}")
    print(f"Image: {images.shape}")
    print(f"Identifier: {identifier}")

# Initialize models
nz = 100  # Size of z latent vector
ngf = 64
ndf = 64
nc = 3  # Number of color channels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(nc, ndf).to(device)
diffusion_model = DiffusionModel(nc).to(device)

# Loss function and optimizers
criterion = nn.MSELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Create directories if they don't exist
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('model'):
    os.makedirs('model')

# Training loop for GAN
num_epochs = 100

for epoch in range(num_epochs):
    for i, (texts, images, identifier) in enumerate(dataloader):
        # Update Discriminator
        netD.zero_grad()
        real_images = images.to(device).clamp(0, 1)  # Ensure images are in [0, 1]
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

        # Save samples and final images for each input image
        if epoch % 10 == 0 and i % 10 == 0:
            vutils.save_image(fake_images, f'output/samples_epoch_{epoch}_id_{identifier[0]}_{current_date}.png', normalize=True)
            if epoch == num_epochs - 1:
                torch.save(fake_images[0].cpu(), f'output/final_image_id_{identifier[0]}_{current_date}.pt')

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

# Save the GAN models
torch.save(netG.state_dict(), 'model/generator.pt')
torch.save(netD.state_dict(), 'model/discriminator.pt')

# Training loop for diffusion model
num_steps = 1000  # Number of steps in the noise schedule
noise_scheduler = CosineScheduler(num_steps)

# Dictionary to accumulate denoised images for averaging
final_images_accum = {}

for epoch in range(num_epochs):
    for i, (texts, images, identifier) in enumerate(dataloader):
        images = images.clamp(0, 1)  # Ensure the images are normalized to the range [0, 1]
        images = images.to(device)
        step = epoch * len(dataloader) + i
        noise_level = noise_scheduler.step(step)
        noisy_images = add_noise(images, noise_level)
        denoised_images = diffusion_model(noisy_images)

        # Calculate loss
        loss = criterion(denoised_images, images)

        # Backward pass and optimization
        diffusion_optimizer.zero_grad()
        loss.backward()
        diffusion_optimizer.step()

        # Accumulate the denoised images
        id_str = identifier[0]
        if id_str not in final_images_accum:
            final_images_accum[id_str] = denoised_images.clone().detach()
        else:
            final_images_accum[id_str] += denoised_images.clone().detach()

        # Save samples for each input image
        if epoch % 10 == 0 and i % 10 == 0:
            vutils.save_image(denoised_images, f'output/diffusion_samples_epoch_{epoch}_id_{id_str}_{current_date}.png', normalize=True)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Diffusion Loss: {loss.item():.4f}")

# Average the accumulated images and save as final images
for id_str, accumulated_image in final_images_accum.items():
    averaged_image = accumulated_image / num_epochs
    vutils.save_image(averaged_image, f'output/diffusion_final_image_id_{id_str}_{current_date}.png', normalize=True)

# Save the diffusion model
torch.save(diffusion_model.state_dict(), 'model/diffusion_model.pt')
