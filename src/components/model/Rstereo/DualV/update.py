import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
    
    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        hx = torch.cat([h,x], dim=1)
        z = F.sigmoid(self.convz1(hx)) 
        r = F.sigmoid(self.convr1(hx)) 
        r_hx = torch.cat([r*h, x], dim=1) 
        q = F.tanh(self.convq1(r_hx))
        h = (1-z)*h + z*q                                   

        hx = torch.cat([h,x], dim=1)
        z = F.sigmoid(self.convz2(hx))
        r = F.sigmoid(self.convr2(hx))
        r_hx = torch.cat([r*h, x], dim=1)
        q = F.tanh(self.convq2(r_hx))
        h = (1-z)*h + z*q

        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes, hidden_dim = 128):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(cor_planes, hidden_dim*2, 1, padding=0) 
        self.convc2 = nn.Conv2d(hidden_dim*2, hidden_dim + hidden_dim//2, 3, padding=1) 
        self.convf1 = nn.Conv2d(1, hidden_dim, 7, padding=3) 
        self.convf2 = nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1) 
        self.conv = nn.Conv2d(hidden_dim*2, hidden_dim-1, 3, padding=1)
    
    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        out = torch.cat([out, flow], dim=1)
        return out


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(cor_planes, hidden_dim=hidden_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim*2)
        self.flow_head = FlowHead(input_dim=hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1), 
                        nn.ReLU(),
                        nn.Conv2d(hidden_dim*2, mask_size**2*9, 1, padding=0),)
    
    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = 0.25*self.mask(net)

        return net, mask, delta_flow


