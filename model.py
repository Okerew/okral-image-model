import os
import torch
import matplotlib.pyplot as plt
import spacy
from train import DiffusionModel, Generator, Discriminator  


# Function to load a pre-trained GAN model
def load_gan_model(generator_path, discriminator_path, nz, ngf, nc, device):
    """
    Load a pre-trained GAN (Generative Adversarial Network) model.

    Args:
        generator_path (str): Path to the pre-trained GAN generator model.
        discriminator_path (str): Path to the pre-trained GAN discriminator model.
        nz (int): Size of the input noise vector.
        ngf (int): Number of filters in the generator.
        nc (int): Number of channels in the generated image.
        device (torch.device): Device to load the model on.

    Returns:
        Generator, Discriminator: The loaded GAN generator and discriminator models.
    """
    generator = Generator(nz, ngf, nc).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    discriminator = Discriminator(nc, ndf=64).to(device)  # Assuming ndf is 64
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    discriminator.eval()

    return generator, discriminator


# Function to load a pre-trained diffusion model
def load_diffusion_model(model_path, nc, ngf, device):
    """
    Load a pre-trained diffusion model.

    Args:
        model_path (str): Path to the pre-trained diffusion model.
        nc (int): Number of channels in the input image.
        ngf (int): Number of filters in the diffusion model.
        device (torch.device): Device to load the model on.

    Returns:
        DiffusionModel: The loaded diffusion model.
    """
    diffusion_model = DiffusionModel(nc, ngf).to(device)
    diffusion_model.load_state_dict(torch.load(model_path, map_location=device))
    diffusion_model.eval()
    return diffusion_model


# Function to generate an image from text using GAN and Diffusion Model
def generate_image_from_text(generator, discriminator, diffusion_model, text, nz):
    """
    Generate an image from text description using GAN and Diffusion Model.

    Args:
        generator (Generator): Pre-trained GAN generator model.
        discriminator (Discriminator): Pre-trained GAN discriminator model.
        diffusion_model (DiffusionModel): Pre-trained diffusion model.
        text (str): Description for the image.
        nz (int): Size of the input noise vector.

    Returns:
        torch.Tensor: The generated image tensor.
    """
    nlp = spacy.load('en_core_web_sm')
    nlp(text)
    device = next(generator.parameters()).device
    noise = torch.randn(1, nz, 1, 1, device=device)

    with torch.no_grad():
        # Generate initial image with GAN
        generated_image = generator(noise).clamp(0, 1)
        # Refine the image with Diffusion Model
        denoised_image = diffusion_model(generated_image).cpu()
        # Discriminate the generated image
        discriminator_output = discriminator(generated_image)

    return denoised_image, discriminator_output


# Interact with the user
def main():
    diffusion_model_path = 'model/diffusion_model.pt'
    generator_path = 'model/generator.pt'
    discriminator_path = 'model/discriminator.pt'  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nz = 100  # Size of z latent vector
    ngf = 64
    nc = 3  # Number of color channels

    generator, discriminator = load_gan_model(generator_path, discriminator_path, nz, ngf, nc, device)
    diffusion_model = load_diffusion_model(diffusion_model_path, nc, ngf, device)

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

        fake_image, discriminator_output = generate_image_from_text(generator, discriminator, diffusion_model, user_input, nz)
        plt.imshow(fake_image.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
