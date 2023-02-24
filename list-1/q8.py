# here includes code for the question 8 of the list

# Referência para o Sobel em imagens
# https://en.wikipedia.org/wiki/Sobel_operator

import cv2
import numpy as np

from q1 import apply_filter

deriv_kernel_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
])

deriv_kernel_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
])

sobel_kernel_x = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
])

sobel_kernel_y = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1],
])

images_filepath = ['./images/q7.jpg', './images/q8_1.jpg', './images/q8_2.jpg']

def get_image(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    # The image is resized for it to be faster
    return cv2.resize(image, dsize=(w // 16, h // 16))

images = [get_image(filepath) for filepath in images_filepath]

def derivative_x(image):
    return apply_filter(image, 3, deriv_kernel_x.flatten())

def derivative_y(image):
    return apply_filter(image, 3, deriv_kernel_y.flatten())

def gradient(deriv_x, deriv_y):
    return np.abs(deriv_x) + np.abs(deriv_y)

def sobel(image):
    der_sobel_x = apply_filter(image, 3, sobel_kernel_x.flatten())
    der_sobel_y = apply_filter(image, 3, sobel_kernel_y.flatten())
    sobel_gradient = np.abs(der_sobel_x) + np.abs(der_sobel_y)
    return sobel_gradient

deriv_imgs = []
for names, image in zip(images_filepath, images):
    deriv_x = derivative_x(image)
    deriv_y = derivative_y(image)
    abs_gradient = gradient(deriv_x, deriv_y)
    deriv_imgs.append([deriv_x, deriv_y, abs_gradient])

final_imgs = []
for images in deriv_imgs:
    for image in images:
        sobel_gradient = sobel(image)
        final_imgs.append(sobel_gradient)

for i, imgs in enumerate(deriv_imgs):
    cv2.imwrite(f'./images-out/deriv_x_{i}_q8.jpg', imgs[0])
    cv2.imwrite(f'./images-out/deriv_y_{i}_q8.jpg', imgs[1])
    cv2.imwrite(f'./images-out/grad_{i}_q8.jpg', imgs[2])

grouped_final_imgs = [final_imgs[i:i+3] for i in range(0, len(final_imgs), 3)]
for i, img in enumerate(grouped_final_imgs):
    cv2.imwrite(f'./images-out/sobel_x_{i}_q8.jpg', img[0])
    cv2.imwrite(f'./images-out/sobel_y_{i}_q8.jpg', img[1])
    cv2.imwrite(f'./images-out/sobel_grad_{i}_q8.jpg', img[2])
