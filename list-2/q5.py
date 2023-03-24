import numpy as np
import cv2
import matplotlib.pyplot as plt

def LoG(src, ksize=3, scale=1.5, ddepth=cv2.CV_16S):
    t_ksize = (ksize, ksize)
    src = cv2.GaussianBlur(src, t_ksize, sigmaX=scale)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src = cv2.Laplacian(src, ddepth, ksize=ksize)
    src = cv2.convertScaleAbs(src)
    return src

if __name__ == '__main__':
    orig_image = cv2.imread('./images/calculator-1.jpg')
    dims = np.array(orig_image.shape[:2]) // 8
    orig_image = cv2.resize(orig_image, dims)
    
    scales = [0, 0.01, 0.05, 0.1, 0.5, 1]
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    calc_index = lambda x: (i // cols, i % cols)
    for i, scale in enumerate(scales):
        image = LoG(orig_image, scale=scale)
        axes[calc_index(i)].set_title(f'scale {scale}')
        axes[calc_index(i)].imshow(image, cmap='gray')
        axes[calc_index(i)].set_aspect('auto')

    plt.show()