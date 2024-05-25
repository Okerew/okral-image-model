import os
import torch
import matplotlib.pyplot as plt
import spacy
from train import DiffusionModel


# Function to load a pre-trained diffusion model
def load_diffusion_model(model_path):
    nc = 3
    ngf = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    diffusion_model = DiffusionModel(nc, ngf).to(device)
    diffusion_model.load_state_dict(torch.load(model_path, map_location=device))
    diffusion_model.eval()
    return diffusion_model


# Function to generate an image from text using Diffusion Model
def generate_image_from_text_diffusion(diffusion_model, text):
    nlp = spacy.load('en_core_web_sm')
    nlp(text)
    nc = 3
    device = next(diffusion_model.parameters()).device
    noise = torch.randn(1, nc, 64, 64, device=device)
    with torch.no_grad():
        denoised_image = diffusion_model(noise).cpu()
    return denoised_image


# Interact with the user
def main():
    diffusion_model_path = 'model/diffusion_model.pt'

    diffusion_model = load_diffusion_model(diffusion_model_path)

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

        fake_image = generate_image_from_text_diffusion(diffusion_model, user_input)
        plt.imshow(fake_image.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
