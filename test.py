from generator import Generator
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 64
gen = Generator().to(device)
noise = torch.randn(1,noise_dim)
img = gen(noise)
img = img.squeeze(0).detach().cpu().numpy()
plt.imshow(img[0])
plt.axis("off")
plt.show()
