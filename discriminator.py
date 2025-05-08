from torch import nn

class Discriminator(nn.Module):
    def __init__(self,img_dim,hidden_dim=256):
        super().__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(img_dim, hidden_dim),
            self.make_disc_block(hidden_dim,1,final_layer=True)
            )
        
    def make_disc_block(self,input_channels,output_channels,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.LeakyReLU(0.1)
                )
        else:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.Sigmoid()
                )
        
    def forward(self,x):
        return self.disc(x)