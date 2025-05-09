from torch import nn

class Generator(nn.Module):
    def __init__(self,img_dim,z_dim=64,hidden_dim=256):
        super().__init__()
        self.gen = nn.Sequential(
                self.get_gen_block(z_dim, hidden_dim),
                self.get_gen_block(hidden_dim, hidden_dim*2),
                self.get_gen_block(hidden_dim*2, hidden_dim*4),
                self.get_gen_block(hidden_dim*4, hidden_dim*8),
                self.get_gen_block(hidden_dim*8, img_dim,final_layer=True)
            )
        
    def get_gen_block(self,input_channels,output_channels,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True))
        else:
            return nn.Sequential(
                nn.Linear(input_channels,output_channels),
                nn.Sigmoid())
        
    def forward(self,noise):
        return self.gen(noise)