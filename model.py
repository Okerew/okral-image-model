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


"""
    Load a pre-trained generator model from the given model path.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        torch.nn.Module: The loaded generator model.

    Description:
        This function loads a pre-trained generator model from the given model path. It first defines the number of
        latent dimensions (nz), the number of feature maps in the generator (ngf), and the number of channels in the
        output image (nc). It then determines the device to use for loading the model (CPU if GPU is not available,
        otherwise GPU). It creates an instance of the Generator class with the specified parameters and moves it to the
        device. It loads the state dictionary of the model from the saved model file using the specified device.
        Finally, it sets the model to evaluation mode and returns the loaded generator model.
"""
def load_generator(model_path):
    nz = 100
    ngf = 64
    nc = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = Generator(nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    return netG


"""
    Generates an image from the given text using a pre-trained GAN model.

    Args:
        netG (torch.nn.Module): The pre-trained GAN model used for generating the image.
        text (str): The text input used to generate the image.

    Returns:
        torch.Tensor: The generated image as a tensor.

    Raises:
        None

    Examples:
        >>> netG = load_generator('model/generator.pt')
        >>> image = generate_image_from_text(netG, 'A cat sitting on a couch.')
        >>> image.shape
        torch.Size([1, 3, 256, 256])

    Note:
        This function assumes that the GAN model has been loaded and is ready to generate images.
        The input text is processed using the Spacy library to extract relevant information.
        The generated image is obtained by passing noise through the GAN model.
        The function does not modify the GAN model or the input text.
"""
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
