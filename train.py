import argparse
import torch.optim as optim
import torch
from loss import Loss
from generator import Generator
from discriminator import Discriminator
from torch import nn

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms 


parser = argparse.ArgumentParser()
parser.add_argument("--epochs",type=int,default=100,help="Epoch sayısı")
parser.add_argument("--batch_size",type=int,default=32,help="Batch sayısı")
parser.add_argument("--lr",type=float,default=1e-3,help = "öğrenme oranı")
parser.add_argument("--noise_dim",type=int,default=64,help = "gürültü vektörü boyutu")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator().to(device)
disc = Discriminator().to(device)
loss = Loss()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5))])
dataset = MNIST(root = "dataset/", transform = transform, download = True) 
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle = True)

opt_gen = optim.Adam(gen.parameters(), lr = args.lr)
opt_disc = optim.Adam(disc.parameters(), lr = args.lr)
criterion = nn.BCELoss()

best_loss = float("inf")

for epoch in range(args.epochs):
    for batch_idx,(img,label) in enumerate(loader):
        
        cur_batch_size = img.size(0)
        real = img.to(device)
        
        noise = torch.randn(cur_batch_size, args.noise_dim).to(device)

        opt_disc.zero_grad()
        disc_loss = loss.get_disc_loss(gen,disc,real,noise,criterion,device)
        disc_loss.backward()
        opt_disc.step()
        
        opt_gen.zero_grad()
        gen_loss = loss.get_gen_loss(gen,disc,noise,criterion,device)
        gen_loss.backward()
        opt_gen.step()
        
        if gen_loss.item() < best_loss :
            best_loss = gen_loss.item()
            torch.save(gen.state_dict(),"model.pt")
            
        if epoch%10 == 0:
            print(f"epoch : {epoch} - generator_loss : {gen_loss} - discriminator_loss : {disc_loss}")
