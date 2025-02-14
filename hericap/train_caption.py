import os
import hydra
import random
import numpy as np
import multiprocessing
from omegaconf import DictConfig

from datasets.caption.field import TextField
from datasets.caption.coco import build_coco_dataloaders
from datasets.caption.metrics import PTBTokenizer, Cider
from models.caption import Transformer
from models.caption.detector import build_detector
from tools.extract_features import extract_vis_features
from utils.cap_scheduler import CosineLRScheduler

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from engine.caption_engine import *
import shutil

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


def main(gpu, config):


    torch.backends.cudnn.enabled = False
    #         0                   8                      
    rank = config.exp.rank * config.exp.ngpus_per_node + gpu
    print(f"----- rank :{rank} ----gpu:{gpu} -----Initial")
    dist.init_process_group('nccl', 'env://', rank=rank, world_size=config.exp.world_size)

    torch.manual_seed(config.exp.seed)
    np.random.seed(config.exp.seed)
    random.seed(config.exp.seed)

    
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    # Swin + Project + Deformable DETR
    detector = build_detector(config).to(device) 
    detector.load_state_dict(torch.load(config.model.detector.checkpoint)['model'], strict=False)
    # Grid Feature + Cap_generator
    model = Transformer(detector=detector, config=config)
    model = model.to(device)

    # print("model:",model)
    # for name, param in model.named_parameters():
    #     print(f"Name: {name}, Shape: {param.shape}")
    
    start_epoch = config.exp.start_epoch
    best_cider_val = 0.0
    best_cider_test = 0.0


    if start_epoch < config.optimizer.freezing_xe_epochs:  
        if getattr(config.optimizer, 'freeze_backbone', False): # Ture     
            print("----------Freezing Swin---------")
            for n, p in model.named_parameters():
                if 'backbone' in n:
                    p.requires_grad = False

        if getattr(config.optimizer, 'freeze_detector', False): # Ture     
            for n, p in model.named_parameters():
                if 'detector' in n:
                    p.requires_grad = False
            print("----------Freezing Swin + Project + Deformable DETR---------")
        # else:
        #     extract_vis_features(detector, config, device, rank)
 
    

    model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
    optimizers = build_optimizers(model, config, mode='xe')                             
    writer = SummaryWriter(log_dir='tensorboard') if rank == 0 or rank == 1 else None  # tensorboard


    if start_epoch < config.optimizer.freezing_xe_epochs \
        and not getattr(config.optimizer, 'freeze_backbone', False):
 
        model.module.cached_features = True 
        dataloaders, samplers, device = build_coco_dataloaders(config, mode= 'load hdf5', device=device)
        
    else:  #   freezing_xe_epochs =0 或 freeze_backbone = True  导致 use_hdf5_feat = False
        model.module.cached_features = False
        dataloaders, samplers, device = build_coco_dataloaders(config, mode='load image', device=device)  
        
    

    text_field = TextField(vocab_path=config.dataset.vocab_path)
    train_dataset = dataloaders['train'].dataset
    cider = Cider(PTBTokenizer.tokenize([e.text for e in train_dataset.examples]))
    tokenizer = multiprocessing.Pool(8)  #config.optimizer.num_workers
    # print("len(dataloaders['train'];",len(dataloaders['train']))

    scheduler = CosineLRScheduler(
        optimizers['model'],
        num_epochs=config.optimizer.freezing_xe_epochs + config.optimizer.finetune_xe_epochs,
        num_its_per_epoch=len(dataloaders['train']),
        init_lr=config.optimizer.xe_lr,
        min_lr=config.optimizer.min_lr,
        warmup_init_lr=config.optimizer.warmup_init_lr,
    )

   
    fr_xe_epochs = config.optimizer.freezing_xe_epochs 
    fr_sc_epochs = fr_xe_epochs + config.optimizer.freezing_sc_epochs  
    ft_xe_epochs = fr_sc_epochs + config.optimizer.finetune_xe_epochs 
    ft_sc_epochs = ft_xe_epochs + config.optimizer.finetune_sc_epochs  
    total_epochs = ft_sc_epochs 
    
    for epoch in range(max(0, start_epoch), total_epochs):
        
         
        if epoch < fr_xe_epochs:
            phase = 'fr_xe'  
        if fr_xe_epochs <= epoch < fr_sc_epochs:
            phase = 'fr_sc'  

        if fr_sc_epochs <= epoch < ft_xe_epochs:
            phase = 'ft_xe'  
        if ft_xe_epochs <= epoch < ft_sc_epochs:
            phase = 'ft_sc'  

 
        if (phase == 'ft_xe' or phase == 'ft_sc') and dataloaders['train'].dataset.image_field.use_hdf5_feat:   
            model.module.cached_features = False
            dataloaders, samplers, device  = build_coco_dataloaders(config, mode='load image', device=device)
        
 
        if (phase == 'fr_xe' or phase == 'ft_xe') and optimizers['mode'] == 'sc':  
            optimizers = build_optimizers(model, config, mode='xe')
        if (phase == 'fr_sc' or phase == 'ft_sc') and optimizers['mode'] == 'xe':
            optimizers = build_optimizers(model, config, mode='sc')

 

        print(f"Train: rank={rank}, epoch={epoch}, phase={phase}")
 
        if phase == 'fr_xe' or phase == 'ft_xe':     
            train_res = train_xe(
                model,
                dataloaders,
                optimizers=optimizers,
                text_field=text_field,
                epoch=epoch,
                rank=rank,
                config=config,
                scheduler=scheduler,
                writer=writer,
                device=device,   # !!!!!!!!!!!!!!!
            )
            samplers['train'].set_epoch(epoch)
 
        elif phase == 'fr_sc' or phase == 'ft_sc':
            
            # for debug
            target_path = os.path.join(os.getcwd(),"checkpoint_best_valid.pth")
            if not os.path.exists(target_path):
                checkpoint = torch.load("/gemini/pretrain2/checkpoint_best_valid.pth", map_location='cpu')
            else:
                checkpoint = torch.load("checkpoint_best_valid.pth", map_location='cpu') 
                

 
            missing, unexpected = model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Start self-critical optimization: missing={len(missing)}, unexpected={len(unexpected)}")
            train_res = train_sc(
                model,
                dataloaders,
                optimizers=optimizers,
                cider=cider,
                text_field=text_field,
                tokenizer_pool=tokenizer,
                device=device,
                epoch=epoch,
                rank=rank,
                config=config,
                writer=writer,
            )
            samplers['train_dict'].set_epoch(epoch)

 
        if rank == 0:
            best_cider_val = evaluate_metrics(
                model,
                optimizers,
                dataloader=dataloaders['valid_dict'],
                text_field=text_field,
                epoch=epoch,
                split='valid',
                config=config,
                train_res=train_res,
                writer=writer,
                best_cider=best_cider_val,
                which=phase,
                scheduler=scheduler,
                device=device,
            )
 
        if rank == 1:
            best_cider_test = evaluate_metrics(
                model,
                optimizers,
                dataloader=dataloaders['test_dict'],
                text_field=text_field,
                epoch=epoch,
                split='test',
                config=config,
                train_res=train_res,
                writer=writer,
                best_cider=best_cider_test,
                which=phase,
                scheduler=scheduler,
                device=device,
            )
 
        if rank == 0:
            save_checkpoint(
                model,
                optimizers,
                epoch=epoch,
                scores=[],
                best_ciders=[0, 0],
                config=config,
                filename=f'checkpoint_{phase}.pth',
                scheduler=scheduler,
            )
            if epoch >= 15:
                save_checkpoint(
                    model,
                    optimizers,
                    epoch=epoch,
                    scores=[],
                    best_ciders=[0, 0],
                    config=config,
                    filename=f'checkpoint_{epoch}.pth',
                    scheduler=scheduler,
                )

        torch.distributed.barrier()


@hydra.main(config_path="configs/caption", config_name="coco_config")
def run_main(config: DictConfig) -> None:

    print("-----config-----:\n", config)
 
    #                    8
    mp.spawn(main, nprocs=config.exp.ngpus_per_node, args=(config,))
    # mp.spawn(main, nprocs=0, args=(config,))


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6688"
    run_main()

