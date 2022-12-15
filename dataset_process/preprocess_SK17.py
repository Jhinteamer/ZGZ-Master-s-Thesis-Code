import os
import sys
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def main():
    img_size_h = 224
    img_size_w = 160

    path_train = glob('C:/Dataset/SK17/ISIC-2017_Training_Data/ISIC-2017_Training_Data/*')
    path_train_mask = 'C:/Dataset/SK17/ISIC-2017_Training_Part1_GroundTruth/'

    path_valid = glob('C:/Dataset/SK17/ISIC-2017_Validation_Data/*')
    path_valid_mask = 'C:/Dataset/SK17/ISIC-2017_Validation_Part1_GroundTruth/'

    path_test = glob('C:/Dataset/SK17/ISIC-2017_Test_v2_Data/*')
    path_test_mask = 'C:/Dataset/SK17/ISIC-2017_Test_v2_Part1_GroundTruth/'

    os.makedirs('inputs/SK17_%d/train_images' % img_size_h, exist_ok=True)
    os.makedirs('inputs/SK17_%d/train_masks' % img_size_h, exist_ok=True)

    os.makedirs('inputs/SK17_%d/valid_images' % img_size_h, exist_ok=True)
    os.makedirs('inputs/SK17_%d/valid_masks' % img_size_h, exist_ok=True)

    os.makedirs('inputs/SK17_%d/test_images' % img_size_h, exist_ok=True)
    os.makedirs('inputs/SK17_%d/test_masks' % img_size_h, exist_ok=True)

    # os.path.basename 返回最后的路径string

    print(path_train[0][:-4],len(path_train))
    print(path_valid[0][:-4],len(path_valid))
    print(path_test[0][:-4],len(path_test))
    print(os.path.basename(path_train[0][-4:]))
    #sys.exit(0)
    # 三个分别对应了训练、验证、测试
    # 验证文件、改名、改变形状一条龙
    for i in tqdm(range(len(path_train))):
        path = path_train[i]
        if path.endswith('.csv'):
            continue
        elif path.endswith('superpixels.png') or path.endswith('superpixels.jpg'):
            continue
        else:
            os.rename(path, path[:-4] + '.png')
            img_path = path[:-4] + '.png'
            mask_path = os.path.join(path_train_mask, (os.path.basename(path)[:-4] + '_segmentation' + '.png'))

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            img = cv2.resize(img, (img_size_h, img_size_w))
            mask = cv2.resize(mask, (img_size_h, img_size_w))
            cv2.imwrite(os.path.join('inputs/SK17_%d/train_images' % img_size_h,
                      os.path.basename(img_path)), img)
            cv2.imwrite(os.path.join('inputs/SK17_%d/train_masks' % img_size_h,
                      os.path.basename(img_path)), mask.astype('uint8'))

    for i in tqdm(range(len(path_valid))):
        path = path_valid[i]
        if path.endswith('.csv'):
            continue
        elif path.endswith('superpixels.png') or path.endswith('superpixels.jpg'):
            continue
        else:
            os.rename(path, path[:-4] + '.png')
            img_path = path[:-4] + '.png'
            mask_path = os.path.join(path_valid_mask, (os.path.basename(path)[:-4] + '_segmentation' + '.png'))

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            img = cv2.resize(img, (img_size_h, img_size_w))
            mask = cv2.resize(mask, (img_size_h, img_size_w))
            cv2.imwrite(os.path.join('inputs/SK17_%d/valid_images' % img_size_h,
                      os.path.basename(img_path)), img)
            cv2.imwrite(os.path.join('inputs/SK17_%d/valid_masks' % img_size_h,
                      os.path.basename(img_path)), mask.astype('uint8'))

    for i in tqdm(range(len(path_test))):
        path = path_test[i]
        if path.endswith('.csv'):
            continue
        elif path.endswith('superpixels.png') or path.endswith('superpixels.jpg'):
            continue
        else:
            os.rename(path, path[:-4] + '.png')
            img_path = path[:-4] + '.png'
            mask_path = os.path.join(path_test_mask, (os.path.basename(path)[:-4] + '_segmentation' + '.png'))

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            img = cv2.resize(img, (img_size_h, img_size_w))
            mask = cv2.resize(mask, (img_size_h, img_size_w))
            cv2.imwrite(os.path.join('inputs/SK17_%d/test_images' % img_size_h,
                      os.path.basename(img_path)), img)
            cv2.imwrite(os.path.join('inputs/SK17_%d/test_masks' % img_size_h,
                      os.path.basename(img_path)), mask.astype('uint8'))

# mask = 0,255
if __name__ == '__main__':
    main()