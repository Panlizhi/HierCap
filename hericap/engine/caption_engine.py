# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------

import os
import time
import json
import torch
import itertools
import numpy as np
from tqdm import tqdm
from datasets.caption import metrics
from torch.nn import NLLLoss
import torch.distributed as dist
from engine.utils import NestedTensor

 

def build_optimizers(model, config, mode='xe'):

    model = getattr(model, 'module', model) # 获取原始模型


    no_decay = ['bias', 'gamma', 'beta'] # 这些参数不进行权重衰减，weight_decay 是一种正则化技术，其主要作用是防止模型过拟合。
    
    # Grid_net + Cap_generator 权重衰减
    model_parameters = [
        {   # 收集参数： requires_grad=Ture且不属于detector且no_decay中至少有一个存在于参数中。
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' not in n and any(nd in n for nd in no_decay) 
            ],
            'weight_decay_rate': 0.0  # 这部分参数衰退率为0
        },
        {   # 收集参数： requires_grad=Ture且不属于detector且不包含no_decay中所有的参数。
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' not in n and not any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': config.optimizer.weight_decay # 这部分参数衰退率不为0
        },
    ]

    # Swin + Project + Deformable DETR 权重衰减
    backbone_parameters = [
        {   # 收集参数： requires_grad=Ture且属于detector且no_decay中至少有一个存在于参数中。
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' in n and any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': 0.0
        },
        {   # 收集参数： requires_grad=Ture且属于detector且不包含no_decay中所有的参数。
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' in n and not any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': config.optimizer.weight_decay
        },
    ]

    optimizers = {
        'model': # Grid_net + Cap_generator 学习率
            torch.optim.Adam(
                model_parameters,
                lr=getattr(config.optimizer, f'{mode}_lr', config.optimizer.sc_lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
            ),
        'backbone': # Swin + Project + Deformable DETR 学习率
            torch.optim.Adam(
                backbone_parameters,
                lr=getattr(config.optimizer, f'{mode}_backbone_lr', config.optimizer.sc_backbone_lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
            ),
        'mode':
            mode
    }
    return optimizers


def gather_result(value):
    if isinstance(value, torch.Tensor):
        torch.distributed.all_reduce(value, async_op=False)  # compute the sum
        value.mul_(1.0 / torch.distributed.get_world_size())  # compute the avg
    return value


def save_checkpoint(
    model,
    optimizers,
    epoch,
    scores,
    best_ciders,
    config=None,
    filename='checkpoint_last.pth',
    scheduler=None,
):
    torch.save(
        {
            "state_dict": model.module.state_dict(),
            "optim_model": optimizers['model'].state_dict(),
            "optim_backbone": optimizers['backbone'].state_dict(),
            "scores": scores,
            "best_ciders": best_ciders,
            "epoch": epoch,
            "exp_name": "" if config is None else config.exp.name,
            "scheduler": [] if scheduler is None else scheduler.state_dict(),
        }, filename)


def log_epoch(config, writer, epoch, train_res, split, scores, which='ft_xe'):
    """For better logging and viewing the log file.
    Run the command in terminal: 
    >>> column -t -s, result.csv
    """
    head = 'exp, backbone, imsize, resize, raug, epoch, split, cider, B1, B4, R, M, B2, B3, t-loss, t-reward, b-reward, which, v-loss'

    if epoch == 0 and not os.path.exists('result.csv'):
        with open('result.csv', 'w') as f:
            f.write(head + '\n')

    with open('result.csv', 'a') as f:
        text = f'{config.exp.name.split("/")[-1]}, '
        backbone = 'B-'
        backbone += 'VG' if os.path.exists(config.model.detector.checkpoint) else 'IM'
        text += f'{backbone}, '
        text += f'{config.dataset.transform_cfg.size[0]}_{config.dataset.transform_cfg.size[1]}, '
        text += f'{config.dataset.transform_cfg.resize_name}, {config.dataset.transform_cfg.randaug}, '
        text += f'{epoch}, {split:<5}, '
        text += f'{scores["CIDEr"]*100:3.2f}, {scores["BLEU"][0]*100:3.2f}, '
        text += f'{scores["BLEU"][3]*100:3.2f}, {scores["ROUGE"]*100:3.2f}, '
        text += f'{scores["METEOR"]*100:3.2f}, {scores["BLEU"][1]*100:3.2f}, {scores["BLEU"][2]*100:3.2f}, '
        text += f'{train_res["loss"]:2.2f}, {train_res["reward"]:2.2f}, {train_res["reward_baseline"]:2.2f}, '
        text += f'{which}, {train_res["val_loss"]:1.2f}'
        f.write(text + '\n')
        print(text)

    writer.add_scalar(f'{split}_cider', scores['CIDEr'], epoch)
    writer.add_scalar(f'{split}_bleu1', scores['BLEU'][0], epoch)
    writer.add_scalar(f'{split}_bleu4', scores['BLEU'][3], epoch)
    writer.add_scalar(f'{split}_meteor', scores['METEOR'], epoch)
    writer.add_scalar(f'{split}_rouge', scores['ROUGE'], epoch)

    writer.add_scalar(f'train_loss', train_res['loss'], epoch)
    writer.add_scalar(f'train_reward', train_res['reward'], epoch)
    writer.add_scalar(f'train_reward_baseline', train_res['reward_baseline'], epoch)


def evaluate_metrics(
    model,
    optimizers,
    dataloader,
    text_field,
    epoch=0,
    split='test',
    config=None,
    train_res=None,
    writer=None,
    best_cider=None,
    which='ft_xe',
    scheduler=None,
    log_and_save=True,
    device=None,
):
    model.eval()
    gen, gts = {}, {}

    counter = 0
    times = []
    with tqdm(desc=f'Epoch {epoch} - evaluation on {split}', unit='it', total=len(dataloader)) as pbar:

        results = []
        for it, batch in enumerate(iter(dataloader)):
            batch['samples'] = batch['samples'].to(device)
            counter += 1
            start_it = time.time()
            with torch.no_grad():
                out, _ , _ = model(
                    batch['samples'],
                    seq=None,
                    use_beam_search=True,
                    max_len=config.model.beam_len,
                    eos_idx=config.model.eos_idx,
                    beam_size=config.model.beam_size,
                    out_size=1,
                    return_probs=False,
                )
            torch.cuda.synchronize()
            end_it = time.time()
            times.append(end_it - start_it)

            if 'samples' in batch and not isinstance(batch['samples'], dict):
                bs = batch['samples'].tensors.shape[0]
            else:
                bs = batch['samples']['reg_feat'].shape[0]
            if it % 100 == 0:
                print(
                    f"Number of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times)/counter:0.5f}s"
                )

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(batch['captions'], caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen[f'{it}_{i}'] = [gen_i]
                gts[f'{it}_{i}'] = gts_i
                res = {'image_id': batch['image_id'][i], 'caption': gen_i}
                results.append(res)
            pbar.update()

    avg_time = sum(times) / counter
    print(f"Epoch: {epoch} iters: {counter}\nTotal time per 1 batch: {avg_time:0.5f}s")
    gts = metrics.PTBTokenizer.tokenize(gts)
    gen = metrics.PTBTokenizer.tokenize(gen)
    scores, _ = metrics.compute_scores(gts, gen)
    print(f'Epoch {epoch}: {split} scores: ' + str(scores) + '\n')

    if log_and_save:
        with open('result.txt', 'a') as f:
            f.write(f'Epoch {epoch}: {split} scores: ' + str(scores) + '\n')
        log_epoch(config, writer, epoch, train_res, split=split, scores=scores, which=which)

        if scores['CIDEr'] >= best_cider:
            best_ciders = (scores['CIDEr'], 0) if split == 'valid' else (0, scores['CIDEr'])
            save_checkpoint(
                model,
                optimizers=optimizers,
                epoch=epoch,
                scores=scores,
                best_ciders=best_ciders,
                config=config,
                filename=f'checkpoint_best_{split}.pth',
                scheduler=scheduler,
            )
            best_cider = scores['CIDEr']
        return best_cider
    else:
        return scores


def inference_coco_test(
    model,
    dataloader,
    text_field,
    epoch=0,
    split='test',
    config=None,
):
    model.eval()
    gen, gts = {}, {}

    counter = 0
    times = []
    with tqdm(desc=f'Epoch {epoch} - evaluation on {split}', unit='it', total=len(dataloader)) as pbar:

        results = []
        for it, batch in enumerate(iter(dataloader)):
            counter += 1
            start_it = time.time()
            with torch.no_grad():
                out, _ , _= model(
                    batch['samples'],
                    seq=None,
                    use_beam_search=True,
                    max_len=config.model.beam_len,
                    eos_idx=config.model.eos_idx,
                    beam_size=config.model.beam_size,
                    out_size=1,
                    return_probs=False,
                )
            torch.cuda.synchronize()
            end_it = time.time()
            times.append(end_it - start_it)

            if 'samples' in batch:
                bs = batch['samples'].tensors.shape[0]
            elif 'vis_feat' in batch:
                bs = batch['vis_feat'].shape[0]
            if it % 100 == 0:
                print(
                    f"Number of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times)/counter:0.5f}s"
                )

            caps_gen = text_field.decode(out, join_words=False)
            for i, gen_i in enumerate(caps_gen):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                res = {'image_id': batch['image_id'][i], 'caption': gen_i}
                results.append(res)
            pbar.update()

    with open(f'result_{split}.json', 'w') as f:
        json.dump(results, f)


def evaluate_loss(model, dataloader, loss_fn, text_field, epoch, writer, device):
    model.eval()

    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                batch['samples'] = batch['samples'].to(device)
                batch['captions'] = batch['captions'].to(device)

                out = model(batch['samples'], batch['captions'])

                captions_gt = batch['captions'][:, 1:].contiguous()
                out = out[:, :-1].contiguous()

                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

                loss = gather_result(loss)
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    if dist.get_rank() == 0:
        writer.add_scalar('val_loss', val_loss, epoch)
    return val_loss


def train_xe(
    model,
    dataloaders,
    optimizers,
    text_field,
    epoch,
    rank=0,
    config=None,
    scheduler=None,
    writer=None,
    device=None, # !!!!!!!!!!!!!!!
):
    model.train()
    # 如果类别标签中包含无法计算损失值的索引（如填充pad索引, token=1），使损失函数忽略这些索引
    loss_fn = NLLLoss(ignore_index = text_field.vocab.stoi['<pad>'])
    
    if scheduler is not None:
        scheduler.step()
    running_loss = .0
    with tqdm(desc=f'Epoch {epoch} - train', unit='it', total=len(dataloaders['train'])) as pbar:
        
        # dataloaders['train']为：
        # ['samples']    从hdf5读的gri和reg特征  或  加载transform变换后、且尺寸统一的image 
        # ['captions']   1个经过截断、增加占位符、pad符后的caption token (长度一致)
        # ['image_id']   图像索引

        # # 记录开始时间
        # start_dataloaders_time = time.time()

        for it, batch in enumerate(dataloaders['train']):
            
            use_hdf5_feat = batch["use_hdf5_feat"]
            # use_hdf5_feat = True
            # print(f"data initial in device {batch['samples']['gri_feat'].device}")
            
            # end_dataloaders_time = time.time()
            # dataloaders_time = end_dataloaders_time - start_dataloaders_time
            # print(f"dataloaders {it} 初始化时间: {dataloaders_time:.6f} 秒")

            start_time = time.time()

            if use_hdf5_feat: # 未做 gri_feat和 reg_feat 消融控制逻辑
                batch['samples']['gri_feat'] = batch['samples']['gri_feat'].to(device)
                batch['samples']['gri_mask'] = batch['samples']['gri_mask'].to(device)
                batch['samples']['reg_feat'] = batch['samples']['reg_feat'].to(device)
                batch['samples']['reg_mask'] = batch['samples']['reg_mask'].to(device)  
                batch['captions'] = batch['captions'].to(device)
            else: #  use_hdf5_feat = False 从图片加载
                batch['samples'] = batch['samples'].to(device)
                batch['captions'] = batch['captions'].to(device)

            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f" transfer the data from CPU to {device} time: {elapsed_time:.6f} 秒")
            # print(f"data after transfer in device {batch['samples']['gri_feat'].device}")
            # print(f"--- transfer the data from CPU to {device} ---")

            out = model(batch['samples'], batch['captions'])   #  (b_s, seq_len, viocabulary_size)
             
            optimizers['model'].zero_grad()     #  Grid_net + Cap_generator
            optimizers['backbone'].zero_grad()  #  Swin + Project + Deformable DETR
            
            # input:              [  2, /  4, 3392, 811, 409, 11,  8,  4,  30, 58,  39,    4, 1838, 119, /3,  1]
            # 期望输出captions_gt: [ /4, 3392,  811, 409,  11,  8,  4, 30,  58, 39,   4, 1838,  119,  /3,  1]
            # 实际输出out        : [ /4, 3392,  811, 409,  11,  8,  4, 30,  58, 39,   4, 1838,  119,  /3,  1, 1]
            captions_gt = batch['captions'][:, 1:].contiguous()  #  (b_s, seq_len-1)              delte <BOS>=2
            out = out[:, :-1].contiguous()                       #  (b_s, seq_len-1, viocabulary_size)   delte <PAD>

            # text_captions_gt0 = text_field.decode(captions_gt[0], join_words=True)
            # text_out0 = text_field.decode(torch.argmax(out, dim=-1)[0], join_words=True)
            # print("text_captions_gt0:",text_captions_gt0)
            # print("text_out0:",text_out0)
            # assert False, "in /gemini/code/grit/engine/caption_engine.py"


            # captions_gt = captions_gt.to(out.device) # 不把数据传到gpu后，后加的!!!!!!!!!!!!

            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optimizers['model'].step()
            optimizers['backbone'].step()

            loss = gather_result(loss)
            running_loss += loss.item()

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

            if scheduler is not None:
                lr = scheduler.step()
                assert optimizers['model'].param_groups[0]['lr'] == lr, "LR scheduler doesn't work properly."

            if rank == 0:
                writer.add_scalar(
                    'backbone_lr', # Swin + Project + Deformable DETR
                    optimizers['backbone'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train']) + it,
                )
                writer.add_scalar(
                    'model_lr',    # Grid_net + Cap_generator
                    optimizers['model'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train']) + it,
                )
                lr = optimizers['model'].param_groups[0]['lr']
            
            # end_time = time.time()
            # train_time = end_time - start_time
            # print(f" training {it} time: {train_time:.6f} 秒")

            # start_dataloaders_time = time.time()

    val_loss = evaluate_loss(model, dataloaders['valid'], loss_fn, text_field, epoch, writer, device)

    if rank == 0:
        save_checkpoint(
            model=model,
            optimizers=optimizers,
            epoch=epoch,
            scores=[],
            best_ciders=(0, 0),
            config=config,
            filename='checkpoint_last.pth',
            scheduler=scheduler,
        )
    torch.distributed.barrier()

    return {
        'loss': running_loss / len(dataloaders['train']),
        'reward': 0,
        'reward_baseline': 0,
        'val_loss': val_loss,
    }


def train_sc(model,
             dataloaders,
             optimizers,
             cider,
             text_field,
             tokenizer_pool,
             device,
             epoch,
             config,
             rank=0,
             writer=None):
    # Training with self-critical
    # 
    # 参考样本（Reference Sample）：
    #   参考样本通常是通过一种确定性的解码策略生成的，比如贪婪解码（Greedy Decoding）或束搜索（Beam Search）。
    #   参考样本反映了模型在给定输入数据时最可能的输出，它通常具有较高的概率，但可能缺乏多样性。
    #       （Greedy Decoding）是指在每个时间步选择概率最高的词；（Beam Search）是在每个时间步保持多个候选序列，并选择整体得分最高的序列。
    
    # 候选样本（Candidate Sample）：
    #   候选样本是通过一种随机化的解码策略生成的，例如采样解码（Sampling Decoding）。
    #   候选样本旨在探索不同的输出空间，从而引入多样性，可能会包含一些新颖或不常见的表达。
    #       （Sampling Decoding）在每个时间步根据模型给出的概率分布随机选择一个词，而不是总是选择概率最高的词。

    running_reward = .0
    running_reward_baseline = .0
    running_loss = .0
    seq_len = config.model.beam_len # 20
    beam_size = config.model.beam_size # 5
    model.train()

    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloaders['train_dict'])) as pbar:
        for it, batch in enumerate(dataloaders['train_dict']):
            
            if 'samples' in batch:
                if isinstance(batch['samples'], NestedTensor):
                    batch['samples'] = batch['samples'].to(device)
                    b_s = batch['samples'].tensors.shape[0]
                elif 'gri_feat' in batch['samples']:
                    batch['samples']['gri_feat'] = batch['samples']['gri_feat'].to(device)
                    b_s = batch['samples']['gri_feat'].shape[0]
                elif 'reg_feat' in batch['samples']:
                    batch['samples']['reg_feat'] = batch['samples']['reg_feat'].to(device)
                    b_s = batch['samples']['reg_feat'].shape[0]
                elif 'glo_feat' in batch['samples']:
                    batch['samples']['glo_feat'] = batch['samples']['glo_feat'].to(device)
                    b_s = batch['samples']['glo_feat'].shape[0]
            elif 'vis_feat' in batch:
                b_s = batch['vis_feat'].shape[0]
                
            optimizers['model'].zero_grad()
            optimizers['backbone'].zero_grad()
            outs, log_probs, _ = model(
                batch['samples'],
                seq=None,
                use_beam_search=True,
                max_len=config.model.beam_len, #20
                eos_idx=config.model.eos_idx,
                beam_size=config.model.beam_size,
                out_size=beam_size, # 5
                return_probs=False,
            )
            # print(f"----------rank {rank} outs.shape:{outs.shape}")   # (16, 5, 20)   [B, out_size, max_len]   每个image对应生成out_size个候选句子，每个句子长max_len
            # print(f"----------rank {rank} len(batch['captions']):{len(batch['captions'])}")
            
            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))   # 生成的描述       [B, out_size, max_len] -> (16*5, 20) --------------> (80,)候选句子list
            caps_gt = list(itertools.chain(*([c] * beam_size for c in batch['captions'])))  # [c,]   真实描述   (16, 5)--复制beam_size次 -> (80,5)个list
            # caps_gen (80,)    caps_gen  ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"..................."sentence 16 4","sentence 16 5"]
            # lcaps_gt (80,5)   caps_gt : [ ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"],
            #                               ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"]
            #                               ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"]
            #                               ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"]
            #                               ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"]
            #                               ["sentence 21","sentence 22","sentence 23","sentence 24","sentence 25"]
            #                               ....[] 
            #                               ["sentence 16 1","sentence 16 2","sentence 16 3","sentence 16 4", "sentence 16 5"] ]

            caps_gen, caps_gt = tokenizer_pool.map(metrics.PTBTokenizer.tokenize, [caps_gen, caps_gt])  # 对生成描述和真实描述进行分词
            # print(f"----------in SC train, rank {rank} 生成的描述caps_gen:{caps_gen}")
            # print(f"----------in SC train, rank {rank} 真实描述caps_gt::{caps_gt}")
            # 
            # bs * beam_size
            # caps_gen 80 dict   caps_gen: {0: [''], 1: ['a'], 2: ['a a'], 3: ['a a a'], 4: ['a a a a a a a a a a a a a a a a a a a a'], 
            #                               5: [''], 6: ['a'], 7: ['a a'], 8: ['a a a'], 9: ['a a a a a a a a a a a a a a a a a a a a'], 
            #                               10: [''], 11: ['a'], 12: ['a a'], 13: ['a a a'], 14: ['a a a a a a a a a a a a a a a a a a a a'], 
            #                               15: [''], 16: ['a'], 17: ['a a'], 18: ['a a a'], 19: ['a a a a a a a a a a a a a a a a a a a a'], 
            #                               .....
            #                               155: [''], 156: ['a'], 157: ['a a'], 158: ['a a a'], 159: ['a a a a a a a a a a a a a a a a a a a a'] }
            # lcaps_gt 80 dict  重复 caps_gt : {0: ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"],
            #                               1: ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"],
            #                               2: ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"],
            #                               3: ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"],
            #                               4: ["sentence 11","sentence 12","sentence 13","sentence 14","sentence 15"],
            #                               5: ["sentence 21","sentence 22","sentence 23","sentence 24","sentence 25"],
            #                               ....[] 
            #                              158: ["sentence 16 1","sentence 16 2","sentence 16 3","sentence 16 4", "sentence 16 5"] }
            #                              159: ["sentence 16 1","sentence 16 2","sentence 16 3","sentence 16 4", "sentence 16 5"] }

            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32) # 计算CIDEr评分

            # debug

            try:
                tensor_reward = torch.from_numpy(reward).to(device).view(b_s, beam_size)
                # 继续后续操作
            except Exception as e:
                print(f"----------in SC train, rank {rank}  生成的reward shape {reward.shape}")
                print(f"----------in SC train, rank {rank}  生成的reward {reward}")
                print("发生错误:", e)
                        
            reward = torch.from_numpy(reward).to(device).view(b_s, beam_size)     # 将评分转移为PyTorch张量并重塑   torch.Size([16, 5])
            reward_baseline = torch.mean(reward, -1, keepdim=True)                # 计算每个样本的平均奖励作为基线
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)        # 模型在训练过程中会被激励去提高那些低于平均水平的样本的性能，同时保持或提高那些已经高于平均水平的样本的性能。

            loss = loss.mean()
            loss.backward()
            torch.distributed.barrier()

            optimizers['model'].step()
            optimizers['backbone'].step()

            loss = gather_result(loss)
            running_loss += loss.item()

            reward = gather_result(reward.mean())
            running_reward += reward.item()

            reward_baseline = gather_result(reward_baseline.mean())
            running_reward_baseline += reward_baseline.item()

            pbar.set_postfix(loss=running_loss / (it + 1),
                             reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
            if rank == 0:
                writer.add_scalar(
                    'backbone_lr',
                    optimizers['backbone'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train_dict']) + it,
                )
                writer.add_scalar(
                    'model_lr',
                    optimizers['model'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train_dict']) + it,
                )

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    val_loss = evaluate_loss(model, dataloaders['valid'], loss_fn, text_field, epoch, writer, device)
    loss = running_loss / len(dataloaders['train_dict'])
    reward = running_reward / len(dataloaders['train_dict'])
    reward_baseline = running_reward_baseline / len(dataloaders['train_dict'])
    if rank == 0:
        save_checkpoint(
            model=model,
            optimizers=optimizers,
            epoch=epoch,
            scores=[],
            best_ciders=(0, 0),
            config=config,
            filename='checkpoint_last.pth',
            scheduler=None,
        )

    torch.distributed.barrier()

    return {'loss': loss, 'reward': reward, 'reward_baseline': reward_baseline, 'val_loss': val_loss}
