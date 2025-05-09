from torch import nn

class Discriminator(nn.Module):
    def __init__(self,img_dim,hidden_dim=128):
        super().__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(img_dim, hidden_dim*4),
            self.make_disc_block(hidden_dim*4, hidden_dim*2),
            self.make_disc_block(hidden_dim*2, hidden_dim),
            self.make_disc_block(hidden_dim,1,final_layer=True)
            )
        
    def get_disc_block(self,input_channels,output_channels,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.LeakyReLU(0.2)
                )
        else:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                )
        
    def forward(self,x):
        return self.disc(x)