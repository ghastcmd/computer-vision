import cv2
import numpy as np

def scale_image(img, s):
    # Altura e largura da imagem original
    h, w = img.shape[:2]
    
    # Novas dimensões da imagem
    new_h, new_w = s*h, s*w
   
    scaled_img = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
    
    # Gerando a nova imagem de maior resolução com a abordagem de vizinho mais próximo
    for i in range(new_h):
        for j in range(new_w):
            ii, jj = int(i/s), int(j/s)
            scaled_img[i,j] = img[ii,jj]
    
    return scaled_img

img = cv2.imread("q6.jpg")

# fatores
s1 = 2
s2 = 3
s3 = 5

scaled_image1 = scale_image(img, s1)
scaled_image2 = scale_image(img, s2)
scaled_image3 = scale_image(img, s3)

cv2.imwrite('./images-out/scaled_q6_1.jpg', scaled_image1)
cv2.imwrite('./images-out/scaled_q6_2.jpg', scaled_image2)
cv2.imwrite('./images-out/scaled_q6_3.jpg', scaled_image3)