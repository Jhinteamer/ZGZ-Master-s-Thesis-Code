import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as albu
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

# TODO
import arch_test.UNetWithResnet50Encoder,arch_test.AGSandSA,arch_test.AGSandSAP1,arch_test.AGSandSAP2,arch_test.Unet
import arch_test.AGSandCBAM,arch_test.UnetandAG,arch_test.SingleAG,arch_test.AGandSAori,arch_test.SA,arch_test.SAP2

import archs
import losses
from dataset import Dataset
from metrics import iou_score,pixel_accuracy,sensitivity,specificity,precision,F1_Score
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():

    parser = argparse.ArgumentParser()
    # TODO
    parser.add_argument('--name', default='SK17_224_UNetWithSAResnet50EncoderandAGsSAP2_woDS',
    help = 'model name')

    args = parser.parse_args()

    return args


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'pixel accuracy': AverageMeter(),
                  'sensitivity': AverageMeter(),
                  'specificy': AverageMeter(),
                  'precision': AverageMeter(),
                  'F1 score': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            # 统计iou时只用最后一张图
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)



        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'pixel accuracy': AverageMeter(),
                  'sensitivity': AverageMeter(),
                  'specificy': AverageMeter(),
                  'precision': AverageMeter(),
                  'F1 score': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                # Deep_sup暂时不统计其他变量
            else:
                output = model(input)
                loss = criterion(output, target)

                ac = pixel_accuracy(output, target)
                se = sensitivity(output, target)
                sp = specificity(output, target)
                pc = precision(output, target)
                f1 = F1_Score(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['pixel accuracy'].update(ac, input.size(0))
            avg_meters['sensitivity'].update(se, input.size(0))
            avg_meters['specificy'].update(sp, input.size(0))
            avg_meters['precision'].update(pc, input.size(0))
            avg_meters['F1 score'].update(f1, input.size(0))


            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('pixel accuracy', avg_meters['pixel accuracy'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('pixel accuracy', avg_meters['pixel accuracy'].avg),
                        ('sensitivity', avg_meters['sensitivity'].avg),
                        ('specificy', avg_meters['specificy'].avg),
                        ('precision', avg_meters['precision'].avg),
                        ('F1 score', avg_meters['F1 score'].avg)])


def main():
    # 文件配置
    args = parse_args()

    # 配置文件的所在位置
    with open('parameters/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 这部分暂时没啥用（跟深监督有关，我的模型没有深监督）
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    # 输出配置
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    # 将配置文件写入models
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # TODO
    # valid or test
    tes = 'valid'

    # create model
    # 比较蠢的方法
    model = arch_test.UNetWithResnet50Encoder.UNetWithResnet50Encoder(config['num_classes'],
                                                                       config['input_channels'],
                                                                       config['deep_supervision'])
    try:
        if config['arch'] == 'UNetWithResnet50Encoder':
            model = arch_test.UNetWithResnet50Encoder.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])
        elif config['arch'] == 'UNetWithSAResnet50EncoderandAGsSA':
            model = arch_test.AGSandSA.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])
        elif config['arch'] == 'UNetWithSAResnet50EncoderandAGsSAP1':
            model = arch_test.AGSandSAP1.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])
        elif config['arch'] == 'UNetWithSAResnet50EncoderandAGsCBAM':
            model = arch_test.AGSandCBAM.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])

        elif config['arch'] == 'UNetplusWithSAResnet50EncoderandAGsSA':
            model = arch_test.UnetandAG.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])

        elif config['arch'] == 'UNetWithSAResnet50EncoderandAG':
            model = arch_test.SingleAG.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])

        elif config['arch'] == 'UNetWithSAResnet50EncoderandAGsSAORI':
            model = arch_test.AGandSAori.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])

        elif config['arch'] == 'UNetWithSAResnet50EncoderandSA':
            model = arch_test.SA.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])

        elif config['arch'] == 'UNetWithSAResnet50EncoderandAGsSAP2':
            model = arch_test.AGSandSAP2.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])

        elif config['arch'] == 'UNetWithSAResnet50EncoderandSAP2':
            model = arch_test.SAP2.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])

        elif config['arch'] == 'UNet':
            model = arch_test.Unet.__dict__[config['arch']](config['num_classes'],
                                                                                config['input_channels'],
                                                                                config['deep_supervision'])

        #UNetWithSAResnet50EncoderandAGsSA
        # UNetWithSAResnet50EncoderandAGsSAP2
        # UNetWithSAResnet50EncoderandSAP2


    except IOError:
        print('cant find model!')

    # model = archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])

    model = model.cuda()
    print("=> creating model %s" % config['arch'])
    # 只保存带梯度的parameter
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # 学习率控制策略
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_test_ids = glob(os.path.join('inputs', config['dataset'], 'train_images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_test_ids]

    img_valid_ids = glob(os.path.join('inputs', config['dataset'], tes + '_images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_valid_ids]


    train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        # transforms.RandomRotate90(),
        # transforms.Flip(),
        # 归一化概率决定执行哪一个
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        albu.Resize(config['input_w'], config['input_h']),
        # transforms.Resize(config['input_w'], config['input_h']),
        transforms.Normalize(),
    ])

    # TODO
    val_transform = Compose([
        albu.Resize(config['input_w'], config['input_h']),
        # transforms.Resize(config['input_w'] , config['input_h'] ),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'train_images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'train_masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], tes + '_images'),
        mask_dir=os.path.join('inputs', config['dataset'], tes + '_masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_accuracy', []),
        ('val_sensitivity', []),
        ('val_specificy', []),
        ('val_precision', []),
        ('val_F1_score', [])
    ])

    best_iou = 0
    best_epoch = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        print('loss %.4f - iou %.4f'
              % (val_log['loss'], val_log['iou']))


        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_accuracy'].append(val_log['pixel accuracy'])
        log['val_sensitivity'].append(val_log['sensitivity'])
        log['val_specificy'].append(val_log['specificy'])
        log['val_precision'].append(val_log['precision'])
        log['val_F1_score'].append(val_log['F1 score'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            # 每次只保存最好的
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            best_epoch = epoch
            print("=> saved best model")
            trigger = 0

        # 早停法
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

    print('training finished, the best miou is Epoch:',best_epoch,' ',best_iou)


if __name__ == '__main__':
    main()
