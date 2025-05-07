from generator import Generator
from discriminator import Discriminator
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator()
disc = Discriminator()

noise = torch.randn(32,64).to(device)
fake_image = gen(noise)
pred = disc(fake_image)

print(fake_image.shape)
print(pred.shape)
