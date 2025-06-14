import torch
import torchvision
from model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("USING DEVICE :",device)

n_label = 10

gen = Generator(n_label).to(device)

gen.eval()

gen.load_state_dict(torch.load("weights/generator.pth"))

sample_n = 64

label = 0
labels = torch.full((sample_n,),label).to(device)

noise = torch.randn((sample_n,100)).to(device)

fake = gen(noise,labels)

img_grid_fake = torchvision.utils.make_grid(fake,normalize=True)

img_grid_fake = torchvision.transforms.ToPILImage()(img_grid_fake)
img_grid_fake.show()