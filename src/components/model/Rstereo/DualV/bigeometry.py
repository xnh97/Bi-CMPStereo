import torch
import torch.nn.functional as F

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]

    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1

    assert torch.unique(ygrid).numel() == 1 and H == 1

    grid = torch.cat([xgrid, ygrid], dim=-1)

    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


class Encoding_Volume:
    def __init__(self, geo_volume0, radius=4, num_levels=2):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume0_pyramid = []

        b, c, d0, h, w = geo_volume0.shape
        #  [b, 8, disp, h, w]   =   [b, h, w, 8, disp]  =   [b*h*w,  8,  1,  disp]
        geo_volume0 = geo_volume0.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d0) 
        
        self.geo_volume0_pyramid.append(geo_volume0)
        for i in range(self.num_levels-1):
            geo_volume0 = F.avg_pool2d(geo_volume0, [1,2], stride=[1,2])
            self.geo_volume0_pyramid.append(geo_volume0)


    def __call__(self, disp):
        r = self.radius
        b, _, h, w = disp.shape
        init_corr_pyramid = []
        geo_feat0_pyramid = []

        dx = torch.linspace(-r, r, 2*r+1)
        dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
        
        for i in range(self.num_levels):
            geo_volume0 = self.geo_volume0_pyramid[i]
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)
            disp_lvl0 = torch.cat([x0,y0], dim=-1)
            geo_feat0 = bilinear_sampler(geo_volume0, disp_lvl0)
            geo_feat0 = geo_feat0.view(b, h, w, -1)
            geo_feat0_pyramid.append(geo_feat0)
        
        geo_feat0 = torch.cat(geo_feat0_pyramid, dim=-1)
        geo_feat0 = geo_feat0.permute(0, 3, 1, 2).contiguous().float() # [N, 2*8*(2r+1), H, W]
        
        return geo_feat0