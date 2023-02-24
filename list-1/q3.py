# here includes code for the question 3 of the list

# Reference for the high-pass and low-pass kernel
# https://dsp.stackexchange.com/questions/49620/how-to-classify-a-kernel-as-low-pass-filter-lpf-or-high-pass-filter-hpf-how

import numpy as np
from PIL import Image

from q1 import apply_filter

images_files = ['./images/q2_1.ppm', './images/q2_2.ppm', './images/q2_3.ppm']

images = [np.array(Image.open(f'{file}').convert('L'), dtype=np.uint8) for file in images_files]

filter_sizes = [3, 6, 9]

kernels = list([
    np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ]),
    np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1,  8,  8, -1, -1],
        [-1, -1,  8,  8, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ]),
    np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1,  8,  8,  8, -1, -1, -1],
        [-1, -1, -1,  8,  8,  8, -1, -1, -1],
        [-1, -1, -1,  8,  8,  8, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]),  
])


images_conv = []
for image in images:
    for size, kernel in zip(filter_sizes, kernels):
        image = image / 255
        image_conv = apply_filter(image, size, kernel.flatten())
        image_conv = image_conv / np.max(image_conv)

        images_conv.append(image_conv)

images_conv_grouped = [images_conv[i:i+3] for i in range(0, len(images_conv), 3)]

out_name = './images-out/high-pass_q3'
kernel_sizes = ['3x3', '6x6', '9x9']
for i, images in zip(range(1, 4), images_conv_grouped):
    for kernel_size, image in zip(kernel_sizes, images):
        Image.fromarray(np.uint8(image * 255)).save(f'{out_name}_{i}_{kernel_size}.ppm')

# É perceptível que quanto maior o filtro de passa-alta, mais grosso ficam as arestas,
# fazendo com que a imagem fique mais com as linhas da imagem mais acentuada, e mais
# perceptível o formato da imagem, de modo que pareça mais com a imagem original.