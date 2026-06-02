import torch
import torch.nn.functional as F

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]

    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1

    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)

    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class Encoding_Volume:
    def __init__(self, geo_volume, radius=4, num_levels=2):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_num = len(geo_volume)
        
        if self.geo_num == 3:
            geo_volume0, geo_volume1, geo_volume2 = geo_volume
        else:
            geo_volume0, geo_volume1 = geo_volume

        self.geo_volume0_pyramid = []
        self.geo_volume1_pyramid = []
        self.geo_volume2_pyramid = []

        b, c, d0, h, w = geo_volume0.shape
        geo_volume0 = geo_volume0.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d0)
        self.geo_volume0_pyramid.append(geo_volume0)

        b, c, d1, h, w = geo_volume1.shape
        geo_volume1 = geo_volume1.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d1)
        self.geo_volume1_pyramid.append(geo_volume1)

        if self.geo_num == 3:
            b, c, d2, h, w = geo_volume2.shape
            geo_volume2 = geo_volume2.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d2) 
            self.geo_volume2_pyramid.append(geo_volume2)        
        
        for i in range(self.num_levels-1):
            geo_volume0 = F.avg_pool2d(geo_volume0, [1,2], stride=[1,2])
            self.geo_volume0_pyramid.append(geo_volume0)

            geo_volume1 = F.avg_pool2d(geo_volume1, [1,2], stride=[1,2])
            self.geo_volume1_pyramid.append(geo_volume1)

            if self.geo_num == 3:
                geo_volume2 = F.avg_pool2d(geo_volume2, [1,2], stride=[1,2])
                self.geo_volume2_pyramid.append(geo_volume2)


    def __call__(self, disp):
        r = self.radius
        b, _, h, w = disp.shape
        geo_feat_pyramid = []

        dx = torch.linspace(-r, r, 2*r+1)
        dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
        
        for i in range(self.num_levels):
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)
            disp_lvl0 = torch.cat([x0,y0], dim=-1)

            geo_volume0 = self.geo_volume0_pyramid[i]
            geo_feat0 = bilinear_sampler(geo_volume0, disp_lvl0)
            geo_feat0 = geo_feat0.view(b, h, w, -1)
            geo_feat_pyramid.append(geo_feat0)

            geo_volume1 = self.geo_volume1_pyramid[i]
            geo_feat1 = bilinear_sampler(geo_volume1, disp_lvl0)
            geo_feat1 = geo_feat1.view(b, h, w, -1)
            geo_feat_pyramid.append(geo_feat1)

            if self.geo_num == 3:
                geo_volume2 = self.geo_volume2_pyramid[i]
                geo_feat2 = bilinear_sampler(geo_volume2, disp_lvl0)
                geo_feat2 = geo_feat2.view(b, h, w, -1)
                geo_feat_pyramid.append(geo_feat2)
        
        geo_feat0 = torch.cat(geo_feat_pyramid, dim=-1)
        geo_feat0 = geo_feat0.permute(0, 3, 1, 2).contiguous().float() 
        
        return geo_feat0