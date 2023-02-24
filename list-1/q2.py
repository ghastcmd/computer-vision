from PIL import Image
import numpy as np

# Mean filter
def mean_filter(image, filter_size, filter_weights):
    # convertendo a imagem em um numpy array
    image = np.array(image)

    # adicionando padding na imagem
    padded_image = np.pad(image, filter_size//2, mode='constant')

    # convertendo os pesos em uma matriz
    filter_matrix = np.reshape(filter_weights, (filter_size, filter_size))

    filtered_image = np.zeros(image.shape)

    # aplicando mean filter através da convolução 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+filter_size, j:j+filter_size]
            filtered_image[i, j] = np.sum(window * filter_matrix) / (np.sum(filter_matrix))

    return filtered_image

# Median filter
def median_filter(image, filter_size, filter_weights):
    image = np.array(image)
    padded_image = np.pad(image, filter_size//2, mode='constant')

    filtered_image = np.zeros(image.shape)

    # median filter através da convolução   
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+filter_size, j:j+filter_size]
            filtered_image[i, j] = np.median(window)

    return filtered_image


filter_size = 3
filter_size = 4
filter_weights = [1, 1, 1, 1, 
                  1, 1, 1, 1, 
                  1, 1, 1, 1,
                  1, 1, 1, 1]

image1 = Image.open('./images/q2_1.ppm').convert('L')
image2 = Image.open('./images/q2_2.ppm').convert('L')
image3 = Image.open('./images/q2_3.ppm').convert('L')

# Aplicando o mean filter nas imagens iniciais e normalizando a imagem
mean_image1 = mean_filter(image1, filter_size, filter_weights)
mean_image1 = (mean_image1 - np.min(mean_image1)) / (np.max(mean_image1) - np.min(mean_image1))

mean_image2 = mean_filter(image2, filter_size, filter_weights)
mean_image2 = (mean_image2 - np.min(mean_image2)) / (np.max(mean_image2) - np.min(mean_image2))

mean_image3 = mean_filter(image3, filter_size, filter_weights)
mean_image3 = (mean_image3 - np.min(mean_image3)) / (np.max(mean_image3) - np.min(mean_image3))

# Aplicando o median filter nas imagens iniciais
median_image1 = median_filter(image1, filter_size, filter_weights)
median_image1 = (median_image1 - np.min(median_image1)) / (np.max(median_image1) - np.min(median_image1))

median_image2 = median_filter(image2, filter_size, filter_weights)
median_image2 = (median_image2 - np.min(median_image2)) / (np.max(median_image2) - np.min(median_image2))

median_image3 = median_filter(image3, filter_size, filter_weights)
median_image3 = (median_image3 - np.min(median_image3)) / (np.max(median_image3) - np.min(median_image3))

# Salvando as imagens
Image.fromarray(np.uint8(mean_image1 * 255)).save('./images-out/mean_q2_1.ppm')
Image.fromarray(np.uint8(mean_image2 * 255)).save('./images-out/mean_q2_2.ppm')
Image.fromarray(np.uint8(mean_image3 * 255)).save('./images-out/mean_q2_3.ppm')
Image.fromarray(np.uint8(median_image1 * 255)).save('./images-out/median_q2_1.ppm')
Image.fromarray(np.uint8(median_image2 * 255)).save('./images-out/median_q2_2.ppm')
Image.fromarray(np.uint8(median_image3 * 255)).save('./images-out/median_q2_3.ppm')