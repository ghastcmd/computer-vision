# here includes code for the question 7 of the list

# Reference for the pyrDown
# https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html

import cv2
import matplotlib.pyplot as plt
import numpy as np

original_image = cv2.imread('./images/q7.jpg')
# Here the image is shrunken because it was too big to visualize in my monitor
original_image = cv2.resize(original_image, (original_image.shape[1]//8, original_image.shape[0]//8))

downsampled_pyramid = [original_image]
for _ in range(3):
    prev_image = downsampled_pyramid[-1]
    new_height = prev_image.shape[0] // 2
    new_width = prev_image.shape[1] // 2
    result = cv2.resize(prev_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    downsampled_pyramid.append(result)

gaussian_pyramid = [original_image]
for _ in range(3):
    prev_image = gaussian_pyramid[-1]
    new_height = prev_image.shape[0] // 2
    new_width = prev_image.shape[1] // 2
    result = cv2.pyrDown(prev_image, dstsize=(new_width, new_height))
    gaussian_pyramid.append(result)

def get_screen_shape(img):
    h, w, d = img.shape
    return (w + (w // 2), h, d)

def create_formated_picture(pyramid: list) -> np.ndarray:
    # Creating the output array
    formated_images = np.zeros(get_screen_shape(pyramid[0]), np.uint8)
    
    # Filling the major image on the left
    h, w = pyramid[0].shape[:2]
    formated_images[:h,:w,:] = pyramid[0][:,:,:]

    # Filling the adjacent images on the right and bellow
    ph, pw = 0, w
    for image in pyramid[1:]:
        h, w = image.shape[:2]
        formated_images[ph:ph+h,pw:pw+w,:] = image[:,:,:]
        ph += h
    
    return formated_images


downsampled_pyramid_image = create_formated_picture(downsampled_pyramid)
gaussian_pyramid_image = create_formated_picture(gaussian_pyramid)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))

axes[0].imshow(cv2.cvtColor(downsampled_pyramid_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('downsampled_pyramid_q7', fontsize=12)
axes[1].imshow(cv2.cvtColor(gaussian_pyramid_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('gaussian_pyramid_q7', fontsize=12)

plt.show()