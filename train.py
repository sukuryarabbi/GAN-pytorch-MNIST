import argparse
import torch.optim as optim
import torch
from loss import Loss
from generator import Generator
from discriminator import Discriminator
from utils import show_img
from torch import nn

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms 


parser = argparse.ArgumentParser()
parser.add_argument("--epochs",type=int,default=100,help="Epoch sayısı")
parser.add_argument("--batch_size",type=int,default=128,help="Batch sayısı")
parser.add_argument("--lr",type=float,default=3e-4,help = "öğrenme oranı")
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
criterion = nn.BCEWithLogisticLoss()

best_loss = float("inf")
mean_gen_loss = 0.0
mean_disc_loss = 0.0
cur_step = 0
display_step = 500

for epoch in range(args.epochs):
    for batch_idx,(real,_) in enumerate(loader):
        cur_batch_size = len(real)
        real = real.view(cur_batch_size,-1).to(device)
        
        opt_disc.zero_grad()
        disc_loss = loss.get_disc_loss(gen,disc,args.noise_dim,cur_batch_size,real,criterion,device)
        disc_loss.backward()
        opt_disc.step()
        
        opt_gen.zero_grad()
        gen_loss = loss.get_gen_loss(gen,disc,args.noise_dim,cur_batch_size,criterion,device)
        gen_loss.backward()
        opt_gen.step()
        
        mean_disc_loss += disc_loss.item() / display_step
        mean_gen_loss += gen_loss.item() / display_step
        
        if mean_gen_loss < best_loss :
            best_loss = gen_loss.item()
            torch.save(gen.state_dict(),"model.pt")
            
    if cur_step%display_step == 0 and cur_step > 0:
        print(f"Step : {cur_step} - generator_loss : {mean_gen_loss} - discriminator_loss : {mean_disc_loss}")
        fake_noise = torch.randn(cur_batch_size,args.noise_dim).to(device)
        fake = gen(fake_noise)
        show_img(fake)
        show_img(real)
        mean_gen_loss = 0.0
        mean_disc_loss = 0.0
    cur_step+=1
