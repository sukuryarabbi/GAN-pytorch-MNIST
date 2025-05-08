from torch import nn

class Generator(nn.Module):
    def __init__(self,img_dim,z_dim=64,hidden_dim=256):
        super().__init__()
        self.gen = nn.Sequential(
                self.make_gen_block(z_dim, hidden_dim),
                self.make_gen_block(hidden_dim, hidden_dim*2),
                self.make_gen_block(hidden_dim*2, img_dim,final_layer=True)
            )
        
    def make_gen_block(self,input_channels,output_channels,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.LeakyReLU(0.1))
        else:
            return nn.Sequential(
                nn.Linear(input_channels,output_channels),
                nn.Tanh())
        
    def forward(self,x):
        return self.gen(x)