import torch

class Loss:
    def __init__(self):
        pass
    
    def get_gen_loss(self,gen,disc,noise_dim,batch_size,criterion,device):
        noise = torch.randn(batch_size,noise_dim).to(device)
        fake_img = gen(noise)
        pred = disc(fake_img).view(-1)
        target = torch.ones_like(pred)
        gen_loss = criterion(pred,target)
        return gen_loss

    
    def get_disc_loss(self,gen,disc,noise_dim,batch_size,real_image,criterion,device):
        noise = torch.randn(batch_size,noise_dim).to(device)
        fake_image = gen(noise)
        real_pred = disc(real_image).view(-1)
        fake_pred = disc(fake_image).view(-1)
        
        real_target = torch.ones_like(real_pred).to(device)
        fake_target = torch.zeros_like(fake_pred).to(device)
        
        real_loss = criterion(real_pred,real_target)
        fake_loss = criterion(fake_pred,fake_target)
        
        return 0.5*(real_loss+fake_loss)
        