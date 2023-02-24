from PIL import Image
import numpy as np

def apply_filter(image, filter_size: int, filter_weights: list[int]) -> np.ndarray:
    # convertendo a imagem em um numpy array
    image = np.array(image)

    # adicionando padding na imagem
    padded_image = np.pad(image, filter_size//2, mode='constant')

    # convertendo os pesos em uma matriz
    filter_matrix = np.reshape(filter_weights, (filter_size, filter_size))

    # fazendo a convolução
    filtered_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+filter_size, j:j+filter_size]
            filtered_image[i, j] = np.sum(window * filter_matrix)

    return filtered_image


image = Image.open('./images/q1.ppm').convert('L')

# definindo o tamanho do filtro e os pesos
filter_size = 3
filter_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1]

# aplicando o filtro
filtered_image = apply_filter(image, filter_size, filter_weights)

# normalizando a imagem
filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image))

Image.fromarray(np.uint8(filtered_image * 255)).save('./images-out/filtered_q1.ppm')