import os

import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
import cv2

def get_harris_corners_coords(image: np.ndarray, k = 0.05, threshold_corner = 0.001) -> list[tuple[int, int]]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    Ix = cv2.Sobel(gray_image, cv2.CV_64F, dx=1, dy=0)
    Iy = cv2.Sobel(gray_image, cv2.CV_64F, dx=0, dy=1)
    
    Ixx = cv2.GaussianBlur(Ix ** 2, (3, 3), 1)
    Ixy = cv2.GaussianBlur(Ix * Iy, (3, 3), 1)
    Iyy = cv2.GaussianBlur(Iy ** 2, (3, 3), 1)

    detA = Ixx * Iyy - Ixy ** 2
    traceA = Ixx + Iyy
    
    harris_response = detA - k * traceA ** 2
    
    # corners = np.copy(image)
    # edges = np.copy(image)
    
    lambda_max = harris_response.max()
    # lambda_min = harris_response.min()
    
    def neighbors_max(x, y):
        return harris_response[x-1:x+2, y-1:y+2].max()
    
    corners_coords = []
    for i, row in enumerate(harris_response[1:-2]):
        for j, pixel in enumerate(row[1:-2]):
            if pixel >= lambda_max * threshold_corner and pixel >= neighbors_max(i+1, j+1):
                # corners[i, j] = [0, 0, 255]
                corners_coords.append((i+1, j+1))
            # elif pixel <= lambda_min * threshold_edge:
            #     edges[i, j] = [255, 0, 255]
    
    # return corners, edges
    return corners_coords


def concat_imgs(img1, img2) -> np.ndarray:
    # getting indexes and creating a zeroed array
    max_height = max(img1.shape[0], img2.shape[0])
    height_1 = img1.shape[0]
    height_2 = img2.shape[0]
    width_1 = img1.shape[1]
    width_2 = img2.shape[1]
    
    channel = img1.shape[2]
    
    concated_images = np.zeros((max_height, width_1 + width_2, channel), dtype=img1.dtype)

    # setting slices of first half to img1 and second half to img2
    concated_images[0:height_1, 0:width_1, :] = img1[:,:,:]
    concated_images[0:height_2, width_1:width_1 + width_2, :] = img2[:,:,:]
    
    return concated_images

def cvt_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def resize_tuple_scaled(shape, factor):
    return shape[0] // factor, shape[1] // factor

def rectangle_corners(center: tuple[int], diag: int) -> tuple[tuple[int, int]]:
        return (center[0] - diag, center[1] - diag), (center[0] + diag, center[1] + diag)

def draw_all_rectangles(image, coords):
    for coord in coords:
        try:
            image[skimage.draw.rectangle_perimeter(*rectangle_corners(coord, 2))] = [255, 0, 0]
        except:
            pass

if __name__ == '__main__':
    root_path = './images'
    image_filepath_1 = 'calculator-1.jpg'
    image_filepath_2 = 'calculator-2.jpg'
    image_1 = cv2.imread(os.path.join(root_path, image_filepath_1))
    image_2 = cv2.imread(os.path.join(root_path, image_filepath_2))

    image_1 = cv2.resize(image_1, resize_tuple_scaled(image_1.shape, 8))
    image_2 = cv2.resize(image_2, resize_tuple_scaled(image_2.shape, 8))
    
    # getting corner's coordinates
    corner_coords_image_1 = get_harris_corners_coords(image_1)
    corner_coords_image_2 = get_harris_corners_coords(image_2)

    # converting images to print on matplotlib
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

    # drawing rectangles for each corner centroid
    to_draw_image_1 = np.copy(image_1)
    draw_all_rectangles(to_draw_image_1, corner_coords_image_1)

    to_draw_image_2 = np.copy(image_2)
    draw_all_rectangles(to_draw_image_2, corner_coords_image_2)

    # plottig images
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    axes[0].imshow(to_draw_image_1)
    axes[0].set_aspect('auto')

    axes[1].imshow(to_draw_image_2)
    axes[1].set_aspect('auto')
    
    plt.show()