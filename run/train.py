import logging
import os
import torch
from torch.utils.data import DataLoader
from models.EGIInet import EGIInet

#import utils.helpers
import argparse
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from run.test import test_net
from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
from dotenv import load_dotenv
from run import hub_uploader
import wandb

load_dotenv()
shapenet_path = os.getenv("SHAPENET_DATASET_PATH")
dino_path = os.getenv("DINO_PROJECT_PATH")
def train_net(cfg):
    torch.backends.cudnn.benchmark = True

    ViPC_train = ViPCDataLoader(os.path.join(dino_path,'train_list.txt'),
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                category=cfg.TRAIN.CATE)
    train_data_loader = DataLoader(ViPC_train,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.CONST.NUM_WORKERS,
                              shuffle=True,
                              drop_last=True,
                              prefetch_factor=cfg.CONST.DATA_perfetch)

    ViPC_test = ViPCDataLoader(os.path.join(dino_path,'test_list.txt'),
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test', category=cfg.TRAIN.CATE)
    val_data_loader = DataLoader(ViPC_test,
                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                   num_workers=cfg.CONST.NUM_WORKERS,
                                   shuffle=True,
                                   drop_last=True,
                                   prefetch_factor=cfg.CONST.DATA_perfetch)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.TRAIN.CATE, '%s', datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    
    model = EGIInet()#.apply(weights_init_normal)
    model.cuda()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    
    # Create the optimizers
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler - Using only CosineAnnealingLR without warmup
    total_epochs = cfg.TRAIN.N_EPOCHS
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    init_epoch = 0
    best_metrics = float('inf')
    BestEpoch = 0

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Load the scheduler's state to resume correctly
        if 'scheduler' in checkpoint:
             lr_scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            init_epoch = checkpoint['epoch']
        logging.info('Recover complete. Resuming from Epoch %d.' % (init_epoch + 1))

    print('recon_points: ',cfg.DATASETS.SHAPENET.N_POINTS, 'Parameters: ', sum(p.numel() for p in model.parameters()))
    
    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

        model.train()

        total_cd_pc = 0
        total_style = 0
        total_loss = 0

        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        with tqdm(train_data_loader) as t:
            for batch_idx, (view,gt_pc,part_pc,depth,) in enumerate(t):

                partial = part_pc.cuda()#[16,2048,3]
                gt = gt_pc.cuda()#[16,2048,3]
                png = view.cuda()
                depth = depth.cuda()
                
                partial = farthest_point_sample(partial,cfg.DATASETS.SHAPENET.N_POINTS)
                gt = farthest_point_sample(gt,cfg.DATASETS.SHAPENET.N_POINTS)
                
                recon,style_loss=model(partial,png,depth)
                cd = chamfer_sqrt(recon,gt)
                loss_total=cd+style_loss*1e-2
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                cd_pc_item = cd.item() * 1e3
                total_cd_pc += cd_pc_item
                style_item = style_loss.item() * 1e1
                total_style += style_item
                loss_item = loss_total.item() * 1e3
                total_loss += loss_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/style', style_item, n_itr)
                train_writer.add_scalar('Loss/Batch/loss', loss_item, n_itr)
                t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, style_item, loss_item]])
                run.log({"cdc":cd_pc_item,"style":style_item,"loss":loss_item,"epoch":epoch_idx,"learning_rate":optimizer.param_groups[0]['lr']})

        avg_cdc = total_cd_pc / n_batches
        avg_style = total_style / n_batches
        avg_loss = total_loss / n_batches

        # Step the scheduler ONCE per epoch AFTER the training loop
        lr_scheduler.step()
        
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/style', avg_style, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/loss', avg_loss, epoch_idx)
        logging.info(
                '[Epoch %d/%d]  Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS,
                 ['%.4f' % l for l in [avg_cdc, avg_style, avg_loss]]))
        
        
        # # Validate the current model
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        
        
        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0:
            file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            # Save scheduler state and epoch for proper resuming
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }, output_path)
            
            logging.info('Saved checkpoint to %s ...' % output_path)
        
        # This logic should compare against the validation metric
        if cd_eval < best_metrics:
            best_metrics = cd_eval
            BestEpoch = epoch_idx
            # Save the best model checkpoint
            best_file_name = 'ckpt-best.pth'
            best_output_path = os.path.join(cfg.DIR.CHECKPOINTS, best_file_name)
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }, best_output_path)

        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch,best_metrics))
        
    run.finish()
    train_writer.close()
    val_writer.close()
