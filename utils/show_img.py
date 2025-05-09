import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_tensor_img(img_tensor,num_images=25,size=(1,28,28)):
    image_unflat = img_tensor.detach().numpy().view(-1,*size)
    image_grid = make_grid(image_unflat[:num_images],nrow=5)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    plt.show()