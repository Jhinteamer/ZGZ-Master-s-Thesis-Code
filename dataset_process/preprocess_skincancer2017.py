import os
import sys
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def main():
    img_size_h = 224
    img_size_w = 160

    paths1 = glob('C:/Dataset/skincancer17/ISIC-2017_Data_train/*')
    paths2 = glob('C:/Dataset/skincancer17/ISIC-2017_Data_groundtruth_train/*')

    os.makedirs('inputs/skincancer2017_%d/images' % img_size_h, exist_ok=True)
    os.makedirs('inputs/skincancer2017_%d/masks' % img_size_h, exist_ok=True)

    # os.path.basename 返回最后的路径string

    # print(paths1[0])
    # print(os.path.basename(paths1[0]))
    # sys.exit(0)

    for i in tqdm(range(len(paths1))):
        path = paths1[i]

        img = cv2.imread(path)
        mask = np.zeros((img.shape[0], img.shape[1]))
        # for mask_path in glob(os.path.join(path, 'masks', '*')):
        #     mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
        #     mask[mask_] = 1
        maskpath = os.path.join(paths2[i])
        # 这里防止实际图片有偏差所以重进建立
        mask_ = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE) > 127
        mask[mask_] = 1

        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]

        img = cv2.resize(img, (img_size_h, img_size_w))
        mask = cv2.resize(mask, (img_size_h, img_size_w))
        cv2.imwrite(os.path.join('inputs/skincancer2017_%d/images' % img_size_h,
                    os.path.basename(path)), img)
        cv2.imwrite(os.path.join('inputs/skincancer2017_%d/masks' % img_size_h,
                    os.path.basename(path)), (mask * 255).astype('uint8'))

# mask = 0,255
if __name__ == '__main__':
    main()