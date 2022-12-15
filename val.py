import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as albu
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import nn

# TODO
import arch_test.AGSandSA,arch_test.UNetWithResnet50Encoder,arch_test.Unet,arch_test.AGSandCBAM,arch_test.AGSandSAP2,arch_test.AGSandSAP1
import arch_test.AGandSAori,arch_test.SAP2
from dataset import Dataset
from metrics import iou_score,pixel_accuracy,sensitivity,specificity,precision,F1_Score
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    # TODO
    # 这里决定读models的哪个文件
    parser.add_argument('--name', default='SK17_224_UNetWithSAResnet50EncoderandAGsSAP2_woDS',
                        help='model name')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    # 这里通过模型名称设定model
    # TODO

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

        elif config['arch'] == 'UNet':
            model = arch_test.Unet.__dict__[config['arch']](config['num_classes'],
                                                                config['input_channels'],
                                                                config['deep_supervision'])

        elif config['arch'] == 'UNetWithSAResnet50EncoderandAGsCBAM':
            model = arch_test.AGSandCBAM.__dict__[config['arch']](config['num_classes'],
                                                                config['input_channels'],
                                                                config['deep_supervision'])

        elif config['arch'] == 'UNetWithSAResnet50EncoderandAGsSAP2':
            model = arch_test.AGSandSAP2.__dict__[config['arch']](config['num_classes'],
                                                                config['input_channels'],
                                                                config['deep_supervision'])

        elif config['arch'] == 'UNetWithSAResnet50EncoderandAGsSAP1':
            model = arch_test.AGSandSAP1.__dict__[config['arch']](config['num_classes'],
                                                                config['input_channels'],
                                                                config['deep_supervision'])
        elif config['arch'] == 'UNetWithSAResnet50EncoderandAGsSAORI':
            model = arch_test.AGandSAori.__dict__[config['arch']](config['num_classes'],
                                                                  config['input_channels'],
                                                                  config['deep_supervision'])

        elif config['arch'] == 'UNetWithSAResnet50EncoderandSAP2':
            model = arch_test.SAP2.__dict__[config['arch']](config['num_classes'],
                                                                  config['input_channels'],
                                                                  config['deep_supervision'])

    #SK17_224_UNetWithSAResnet50EncoderandSAP2_woDS
    except IOError:
        print('cant find model!')

    model = model.cuda()

    # Data loading code
    # 用哪个模型？
    # TODO
    val = 'valid'

    img_test_ids = glob(os.path.join('inputs', config['dataset'], val + '_images', '*' + config['img_ext']))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_test_ids]


    # 加载训练过的模型
    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    # TODO
    val_transform = Compose([
        albu.Resize(config['input_w'], config['input_h']),
        # transforms.Resize(config['input_w'], config['input_h']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], val + '_images'),
        mask_dir=os.path.join('inputs', config['dataset'], val + '_masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter_iou = AverageMeter()
    avg_meter_pa = AverageMeter()
    avg_meter_se = AverageMeter()
    avg_meter_sp = AverageMeter()
    avg_meter_pc = AverageMeter()
    avg_meter_f1 = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            pa = pixel_accuracy(output, target)
            se = sensitivity(output, target)
            sp = specificity(output, target)
            pc = precision(output, target)
            f1 = F1_Score(output, target)
            avg_meter_iou.update(iou, input.size(0))
            avg_meter_pa.update(pa, input.size(0))
            avg_meter_se.update(se, input.size(0))
            avg_meter_sp.update(sp, input.size(0))
            avg_meter_pc.update(pc, input.size(0))
            avg_meter_f1.update(f1, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output = output > 0.5

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('Pixel Acc: %.4f' % avg_meter_pa.avg)
    print('Sensitivity: %.4f' % avg_meter_se.avg)
    print('Specificity: %.4f' % avg_meter_sp.avg)
    print('Precision: %.4f' % avg_meter_pc.avg)
    print('F1 Score: %.4f' % avg_meter_f1.avg)
    print('IoU: %.4f' % avg_meter_iou.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
