from torch import nn

class Generator(nn.Module):
    def __init__(self,z_dim=64,img_channel=1,hidden_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim,hidden_dim*4),
            self.make_gen_block(hidden_dim*4, hidden_dim*2,kernel_size=4,stride=1),
            self.make_gen_block(hidden_dim*2, hidden_dim),
            self.make_gen_block(hidden_dim,img_channel,kernel_size=4,final_layer=True),            
            )
        
    def make_gen_block(self,input_channels,output_channels,kernel_size=3,stride=2,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU())
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh())
        
    def forward(self,x):
        x = x.view(len(x),self.z_dim,1,1)
        return self.gen(x)