import cv2
import numpy as np
import matplotlib.pyplot as plt


def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv


def gaussian_noise(image):        # 加高斯噪声
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]   # blue
            g = image[row, col, 1]   # green
            r = image[row, col, 2]   # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    dst = cv2.GaussianBlur(image, (15, 15), 0)  # 高斯模糊
    return dst, image


if __name__ == "__main__":
     src = cv2.imread('ISIC_0000471.png')

     # plt.subplot(2, 2, 1)
     # plt.imshow(src)
     # plt.axis('off')
     # plt.title('Offical')

     output, noise = gaussian_noise(src)
     cvdst = cv2.GaussianBlur(src, (15, 15), 0)   # 高斯模糊

     # plt.subplot(2, 2, 2)
     # plt.imshow(noise)
     # plt.axis('off')
     # plt.title('Gaussian Noise')

     # plt.subplot(2, 2, 3)
     plt.imshow(output)
     plt.axis('off')
     # plt.title('Gaussian Blur')

     # plt.subplot(2, 2, 4)
     # plt.imshow(cvdst)
     # plt.axis('off')
     # plt.title('defined blur by opencv')

     plt.savefig("gauss_blur.png")
