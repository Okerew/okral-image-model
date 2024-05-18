import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import spacy

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

# Load the generator model
def load_generator(model_path):
    nz = 100
    ngf = 64
    nc = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = Generator(nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    return netG

# Function to generate an image from user input
def generate_image_from_text(netG, text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    nz = 100
    device = next(netG.parameters()).device
    noise = torch.randn(1, nz, 1, 1, device=device)
    with torch.no_grad():
        fake_image = netG(noise).cpu()
    return fake_image

# Interact with the user
def main():
    model_path = 'model/generator.pt'
    netG = load_generator(model_path)
    while True:
        print("Enter ?help for a list of available commands.")
        user_input = input("Enter a description for the image: ")
        if user_input.lower() == '?quit':
            break
        elif user_input.lower() == '?help':
            print("Available commands:")
            print("?help - Show this help message")
            print("?quit - Quit the program")
            print("?clear - Clear the screen")
            continue
        elif user_input.lower() == '?clear':
            os.system('clear')
            continue

        fake_image = generate_image_from_text(netG, user_input)
        plt.imshow(fake_image.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
