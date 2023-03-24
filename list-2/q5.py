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


def is_max_of_box(z, y, x, grid, ksize=3):
    # getting the size from center
    bounds = (ksize - 1) // 2
    
    # getting the neighbors of z, y, x
    max_of_box = grid[
        z - bounds : z + bounds + 1,
        x - bounds : x + bounds + 1,
        y - bounds : y + bounds + 1,
    ].max()

    return grid[z, x, y] >= max_of_box

def draw_local_max(grid):
    # getting the bounds of grid
    max_z, max_x, max_y = grid.shape[:3]

    ret_grid = [grid[0]]
    for z in range(max_z-2):
        image = cv2.cvtColor(grid[z+1], cv2.COLOR_GRAY2RGB)
        for y in range(max_y-2):
            for x in range(max_x-2):
                if is_max_of_box(z+1, y+1, x+1, grid):
                    image[x+1, y+1] = [0, 255, 0]
        ret_grid.append(image)

    ret_grid.append(grid[-1])

    return ret_grid


if __name__ == '__main__':
    orig_image = cv2.imread('./images/calculator-1.jpg')
    dims = np.array(orig_image.shape[:2]) // 8
    orig_image = cv2.resize(orig_image, dims)
    
    scales = [0.0001, 0.01, 0.1, 0.5, 0.7, 1]
    image_list = []
    for scale in scales:
        image = LoG(orig_image, scale=scale)
        image_list.append(image)

    image_list = np.array(image_list)
    image_list = draw_local_max(image_list)
    
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    calc_index = lambda x: (x // cols, x % cols)
    for i, (image, scale) in enumerate(zip(image_list, scales)):
        axes[calc_index(i)].imshow(image, cmap='gray')
        axes[calc_index(i)].set_title(f'scale {scale}')
        axes[calc_index(i)].set_aspect('auto')

    plt.show()