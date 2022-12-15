import os
import sys
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def main():
    img_size_h = 224
    img_size_w = 160

    os.makedirs('../inputs/SK18_%d/train_images' % img_size_h, exist_ok=True)
    os.makedirs('../inputs/SK18_%d/train_masks' % img_size_h, exist_ok=True)

    os.makedirs('../inputs/SK18_%d/test_images' % img_size_h, exist_ok=True)
    os.makedirs('../inputs/SK18_%d/test_masks' % img_size_h, exist_ok=True)

    path = glob('../inputs/SK18_%d/train_images/*' % img_size_h)
    path_mask = glob('../inputs/SK18_%d/train_masks/*' % img_size_h)

    # test
    print(path[0][:-4], len(path))
    print(os.path.basename(path[0][-4:]))
    print('------------')
    print(path_mask[0][:-4], len(path_mask))
    print(os.path.basename(path_mask[0][-4:]))

    train_nums = (2594 * 8) / 10

    pp = len(path)

    # os.path.basename 返回最后的路径string
    for i in tqdm(range(pp)):
        if i < train_nums:
            continue

        img_name = path[i]
        mask_name = path_mask[i]

        img = Image.open(img_name)
        mask = Image.open(mask_name)

        img.save(os.path.join('../inputs/SK18_%d/test_images' % img_size_h,
                  os.path.basename(img_name)), "png")
        mask.save(os.path.join('../inputs/SK18_%d/test_masks' % img_size_h,
                                os.path.basename(mask_name)), "png")

        os.remove(img_name)
        os.remove(mask_name)


        # cv2.imwrite(os.path.join('../inputs/SK18_%d/train_images' % img_size_h,
        #           os.path.basename(img_name)), img_i)
        # cv2.imwrite(os.path.join('../inputs/SK18_%d/train_masks' % img_size_h,
        #           os.path.basename(mask_name)), mask_i.astype('uint8'))

# mask = 0,255


if __name__ == '__main__':
    main()
