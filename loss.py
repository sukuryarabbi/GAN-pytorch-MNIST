import torch

class Loss:
    def __init__(self):
        pass
    
    def get_gen_loss(self,gen,disc,noise,criterion,device):
        fake_image = gen(noise)
        preds = disc(fake_image)
        target = torch.ones_like(preds).to(device)
        loss = criterion(preds,target)
        return loss
    
    def get_disc_loss(self,gen,disc,real_image,noise,criterion,device):
        fake_image = gen(noise)
        real_pred = disc(real_image)
        fake_pred = disc(fake_image)
        
        real_target = torch.ones_like(real_pred).to(device)
        fake_target = torch.zeros_like(fake_pred).to(device)
        
        real_loss = criterion(real_pred,real_target)
        fake_loss = criterion(fake_pred,fake_target)
        
        return 0.5*(real_loss+fake_loss)
        