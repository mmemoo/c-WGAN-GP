import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self,n_label):
        super().__init__()
        self.embedding = nn.Embedding(n_label,1)

        self.block1 = self.create_block(101,1024,4,1,0) # 4x4
        self.block2 = self.create_block(1025,512,4,2,1) # 8x8
        self.block3 = self.create_block(513,256,4,2,1) # 16x16
        self.block4 = nn.Sequential(nn.ConvTranspose2d(257,1,4,2,1),nn.Tanh()) #32x32

    def create_block(self,in_f,out_f,kernel,stride,pad):
        deconv = nn.ConvTranspose2d(in_f,out_f,kernel,stride,pad)
        batch_norm = nn.BatchNorm2d(out_f)
        relu = nn.ReLU()
        return nn.Sequential(deconv,batch_norm,relu)

    def forward(self,x,labels):
        labels = self.embedding.to("cpu" if device == "mps" else device)(labels.to("cpu" if device == "mps" else device)).to(device)
        labels = labels.view(labels.shape[0],1,1,1)

        x = x.view(-1,100,1,1)
        
        x = torch.cat((x,labels),dim=1)
        
        b,_,_,_ = x.shape

        x = self.block1(x)
        x = torch.cat((x,torch.randn((b,1,4,4)).to(x.device)),1)
        x = self.block2(x)
        x = torch.cat((x,torch.randn((b,1,8,8)).to(x.device)),1)
        x = self.block3(x)
        x = torch.cat((x,torch.randn((b,1,16,16)).to(device)),1)
        x = self.block4(x)
        return x
    
class MiniBatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        b,_,h,w = x.shape
        mbsd = torch.std(x,0,unbiased=False,keepdim=True) + 1e-8
        mbsd = torch.mean(mbsd)
        mbsd = mbsd.expand((b,1,h,w))
        return torch.cat((x,mbsd),1)

class Critic(nn.Module):
    def __init__(self,n_label,leak_val):
        super().__init__()
        self.embedding = nn.Embedding(n_label,32*32)

        self.block1 = nn.Sequential(nn.Conv2d(2,128,4,2,1),nn.LeakyReLU(leak_val)) #16x16
        self.block2 = self.create_block(128,256,4,2,1,leak_val) # 8x8
        self.block3 = self.create_block(256,512,4,2,1,leak_val) # 4x4
        self.block4 = nn.Sequential(MiniBatchStdDev(),nn.Conv2d(513,1,4,1,0)) # 1x1
    
        self.learned_matrix = nn.Parameter(torch.randn((512*4*4)),requires_grad=True)

    def create_block(self,in_f,out_f,kernel,stride,pad,leak_val):
        conv = nn.Conv2d(in_f,out_f,kernel,stride,pad)
        batch_norm = nn.BatchNorm2d(out_f)
        l_relu = nn.LeakyReLU(leak_val)
        return nn.Sequential(conv,batch_norm,l_relu)
    
    def forward(self,x,labels):
        labels = self.embedding.to("cpu" if device == "mps" else device)(labels.to("cpu" if device == "mps" else device)).to(device)
        labels = labels.view(labels.shape[0],1,32,32)

        x = torch.cat((x,labels),1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        mb_features = x.view(x.shape[0],-1) * self.learned_matrix
        mb_features = mb_features.unsqueeze(1) - mb_features.unsqueeze(0)
        mb_features = torch.sqrt((mb_features ** 2).sum(-1) + 1e-8)
        sims = torch.exp(-mb_features)
        sims = sims * (1 - torch.eye(sims.shape[0],sims.shape[1],device=sims.device))
        sims = torch.sum(sims,1).view(-1,1,1,1)
        x = x + sims

        x = self.block4(x)
        return x.view(-1,1)