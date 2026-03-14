import torch
from torch.utils.data import DataLoader
import timm
from dataset import Branch1_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet
from engine_branch1 import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")
DEFAULT_DATA_PATH = str(Path(__file__).resolve().parents[1] / "datasets" / "arcade" / "data" / "vessel")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Branch1')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--work_dir', type=str, default='./work_dir/branch1', help='work directory')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='data path')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True, help='enable mixed precision')
    parser.add_argument('--amp_dtype', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True, help='enable TF32 on Ampere+ GPUs')
    return parser.parse_args()


def main(config, args):
    print('#----------Creating logger----------#')
    config.work_dir = args.work_dir
    config.data_path = os.path.join(args.data_path, '')
    config.batch_size = args.batch_size
    config.gpu_id = args.gpu_id
    config.epochs = args.epochs
    config.num_workers = args.num_workers
    config.amp = args.amp
    config.amp_dtype = args.amp_dtype
    config.grad_accum_steps = max(1, args.grad_accum_steps)

    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    gpu_id = int(config.gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = Branch1_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers,
                              persistent_workers=config.num_workers > 0)
    val_dataset = Branch1_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            persistent_workers=config.num_workers > 0,
                            drop_last=True)

    test_dataset = Branch1_datasets(config.data_path, config, train=False, test=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             persistent_workers=config.num_workers > 0,
                             drop_last=True)

    print('#----------Prepareing Pure VM-UNet----------#')
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()
    model = model.to(device)

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and config.amp_dtype == 'fp16')

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device,
            scaler=scaler,
        )

        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
            device
        )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
            device
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    args = parse_args()
    main(config, args)
