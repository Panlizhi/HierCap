import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.common.swin_model import *
from utils.misc import nested_tensor_from_tensor_list, NestedTensor
from models.detection.det_module import build_det_module_with_config


class Detector(nn.Module):

    def __init__(
        self,
        backbone,
        det_module=None,
        use_gri_feat=True,
        use_reg_feat=True,
        use_glob_feat=True,
        hidden_dim=256,    # 512
    ):
        super().__init__()

        # Swin
        self.backbone = backbone
        self.use_gri_feat = use_gri_feat
        self.use_reg_feat = use_reg_feat
        self.use_glob_feat = use_glob_feat

        # Project + Deformable DETR
        if self.use_reg_feat:
            self.det_module = det_module
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[i], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ) for i in range(len(backbone.num_channels))     # num_channels [2C, 4C, 8C, 1024]
            ])
    
    
    def forward(self, images: NestedTensor):  
        # - images.tensor: batched images, of shape [batch_size x 3 x H x W]
        # - images.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        '''
        input:
        batch['samples']    从hdf5读的gri和reg特征(list)  或  加载transform变换后、且尺寸统一的image(nested_tensor)
        '''

        if isinstance(images, (list, torch.Tensor)): # images 是列表或张量 ---> nested张量
            samples = [img for img in images]
            images = nested_tensor_from_tensor_list(samples)

        x = images.tensors  # RGB input # [B, 3, H, W]
        mask = images.mask  # padding mask [B, H, W]  torch.Size([16, 384, 640])   在 img 有值的地方是 0，补零的地方是 1。

          
        # x:[B, ori_h, ori_w]  --->  [[B, C1, H1, W1], [B, C2, H2, W2], [B, C3, H3, W3], [B, C4, H4, W4]]
        # output:                      --->       swin2            swin3                 swin4              patchmerging4
        # features: [batch, 384, 640]  --->  [[16, 256, 48, 80], [16, 512, 24, 40], [16, 1024, 12, 20], [16, 1024, 6, 10]] 
        features = self.backbone(x)
        
        
        # mask[None]  (1,B,H,W)  [1, 16, 384, 640] 
        masks = [
            F.interpolate(mask[None].float(), size=f.shape[-2:]).to(torch.bool)[0] for l, f in enumerate(features)
        ] 
        # masks.shape  [B, Hinter, Winter]   [[16, 48, 80] ,[16, 24, 40], [16, 12, 20],[16, 6, 10]]

        outputs = {}
        outputs['gri_feat'] = rearrange(features[-1], 'b c h w -> b (h w) c')
        outputs['gri_mask'] = repeat(masks[-1], 'b h w -> b 1 1 (h w)')

        print(f" (h, w) is the dim of the image's gri_feat: {features[-1].shape[2]}, {features[-1].shape[3]} in detector.py")
        
        if self.use_reg_feat:
            # Project
 
            srcs = [self.input_proj[l](src) for l, src in enumerate(features)]
            # Deformable DETR
            # srcs.shape [[16, 512, 48, 80], [16, 512, 24, 40], [16, 512, 12, 20], [16, 512, 6, 10]]
            # hs:  7   [[B, num_queries, C],  [B, num_queries, C],......]  num_queries=150  C=512
            # inter_references:  7  [[B, num_queries, 4],[B, num_queries, 4]........]
            hs, _ , inter_references_out = self.det_module(srcs, masks) 
            
            outputs['reg_feat'] = hs[-1]
            # new_full 新张量初始化为0， reg_mask所有元素都设置为 False  
            outputs['reg_mask'] = hs[-1].data.new_full((hs[-1].shape[0], 1, 1, hs[-1].shape[1]), 0).bool()  
            outputs['reg_point'] = inter_references_out[-1]
 
        
        if self.use_glob_feat:
            #  4 b c h w
            outputs['glo_feat'] = features[-1]  # b c h w    [16, 1024, 6, 10]
            outputs['glo_mask'] = masks[-1]     # b h w      [16, 6, 10]
            # outputs['glo_mask'] = repeat(masks[-1], 'b h w -> b 1 1 (h w)')     # b 1 1 h w     [16, 1, 1, 60]
            # outputs['glo_mask'] = [[repeat(src2, 'b h w -> b 1 1 (h w)')] for l2, src2 in enumerate(masks)]   
            # print(f"Type of outputs['glo_mask'][0]: {type(masks[0])}")
 
            
        return outputs


def build_detector(config):
    pos_dim = getattr(config.model.detector, 'pos_dim', None)  
    
    ######### Swin  
 
    backbone, _ = swin_base_win7_384(
        pretrained=None,
        frozen_stages=config.model.frozen_stages, # 2
        pos_dim=pos_dim,
    )
    
    # if config.model.use_reg_feat = Ture
    #########  Deformable DETR
    det_cfg = config.model.detector
    det_module = build_det_module_with_config(det_cfg) if config.model.use_reg_feat else None

    detector = Detector(
        backbone,               # Swin
        det_module=det_module,  # Deformable DETR
        hidden_dim=config.model.d_model,  # 512
        use_gri_feat=config.model.use_gri_feat,
        use_reg_feat=config.model.use_reg_feat,
        use_glob_feat=config.model.use_glob_feat,
    )
    if os.path.exists(config.model.detector.checkpoint):
        checkpoint = torch.load(config.model.detector.checkpoint, map_location='cpu')
        missing, unexpected = detector.load_state_dict(checkpoint['model'], strict=False) 
        print(f"Loading weights for detector: missing: {len(missing)}, unexpected: {len(unexpected)}.")
    return detector
