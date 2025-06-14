import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as d_utils
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
from model import Generator,Critic

subprocess.Popen(["tensorboard","--logdir=Logs","--reload_multifile=true","--load_fast=false"])

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),transforms.Normalize((0.5),(0.5))])
MNIST = torchvision.datasets.MNIST(".data",True,transform,download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("USING DEVICE :",device)

batch_size = 64
MNIST = d_utils.DataLoader(MNIST,batch_size,True)

epochs = 500
lr = 0.0002
n_crit_step = 5
lambda_ = 10
leak_val = 0.2

n_label = 10

crit = Critic(n_label,leak_val).to(device)
gen = Generator(n_label).to(device)

betas = (0.5,0.999)

crit_optim = torch.optim.Adam(crit.parameters(),lr,betas)
gen_optim = torch.optim.Adam(gen.parameters(),lr,betas)

criterion = nn.BCELoss()

noise_dim = 100

fixed_noise = torch.rand((batch_size,noise_dim)).to(device)
fixed_labels = torch.randint(0,10,(batch_size,)).to(device)
print(fixed_labels)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

f_rl,r_rl,gen_rl,d_rl = 0,0,0,0
n_plot = 0
c_rl,g_rl = 0,0

for epoch in range(1,epochs+1):
    for i,(real,labels) in enumerate(MNIST):
        batch_size = real.shape[0]
        real = real.to(device).view(batch_size,1,32,32)
        
        for j in range(n_crit_step):
            noise = torch.randn((batch_size,noise_dim)).to(device)

            fake = gen(noise,labels)
        
            real_pred = crit(real,labels)
            fake_pred = crit(fake.detach(),labels)
            c_loss = torch.mean(fake_pred,0) - torch.mean(real_pred,0)

            epsilon = torch.rand(1).to(device)
            x_hat = (epsilon * real + (1-epsilon) * fake.detach()).to(device)
            x_hat.requires_grad = True
            
            out = crit(x_hat,labels)

            grad_outs = torch.ones_like(out)

            grads = torch.autograd.grad(out,x_hat,grad_outputs=grad_outs,create_graph=True)[0]
            
            grads = grads.view(batch_size,-1)
            grads_norm = torch.sqrt(torch.sum(grads**2,1))
            penalty = (grads_norm-1)**2
            penalty = torch.mean(penalty)

            c_loss = c_loss + lambda_ * penalty

            crit_optim.zero_grad()
            c_loss.backward()
            crit_optim.step()

            c_rl += c_loss.detach()

        fake_pred = crit(fake,labels)
        gen_loss = -torch.mean(fake_pred)

        g_rl += gen_loss.detach()

        gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()

        with torch.no_grad():
            gen.eval()
            fake = gen(fixed_noise,fixed_labels).to(device)
            div_term = ((epoch-1)*len(MNIST))+i
            print(f"c_loss:{c_rl/(div_term*n_crit_step)},gen_loss:{g_rl/div_term}")
            gen.train()

            torch.save(gen.state_dict(),"generator1.pth")
            torch.save(crit.state_dict(),"discriminator1.pth")

            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(real, normalize=True)

            writer_fake.add_image("Fake Images", img_grid_fake, global_step=n_plot)
            writer_real.add_image("Real Images", img_grid_real, global_step=n_plot)
            n_plot += 1