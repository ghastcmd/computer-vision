# Aplicando o filtro Canny() em uma imagem
import cv2
import numpy as np

img = cv2.imread('./images/q5.ppm')
canny_image = cv2.Canny(img, 150, 250, apertureSize=3)

img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
canny_image = cv2.resize(canny_image, (0, 0), fx=0.5, fy=0.5)

cv2.imshow('img', img)
cv2.imshow('canny_img', canny_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" 
Influencia dos parâmetros threshold1, threshold2 e apertureSize:

### Threshold1
O threshold1 é o parâmetro que indica o limiar mínimo para um pixel ser realmente 
considerado como uma aresta, pixels com um gradiente abaixo desse limiar são descartados. 
Consequentemente, maiores valores para o threshold1 fazem com que menos arestas 
sejam detectadas.

### Threshold2
Já o threshold2, é o parâmetro que indica o limiar máximo, ou seja, qualquer 
pixel acima desse valor será automaticamente considerado como uma aresta e representado
com linhas fortes. Isso quer dizer que se diminuirmos o threshold2, mais arestas fortes 
seriam consideradas por motivo dos valores dos pixels estarem acima desse limiar máximo,
aumentando a quantidade de detalhes na imagem e possivelmente gerando mais ruido.


### Aperture_size
O apertureSize é um parametro opicional que tem seu valor predefinido em "3", 
também podendo ser "5" ou "7". Ao aumentarmos o apertureSize 
mais detalhes na imagem serão detectados, o que é útil para detectar bordas mais finas ou suaves, 
e se diminuirmos o apertureSize, menos detalhes na imagem serão detectados. 
"""
