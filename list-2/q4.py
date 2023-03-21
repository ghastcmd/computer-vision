import os

import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
import cv2

def get_harris_corners_coords(gray_image: np.ndarray, k = 0.05, threshold = 0.001) -> list[tuple[int, int]]:
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
            if pixel >= lambda_max * threshold and pixel >= neighbors_max(i+1, j+1):
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

def compute_sift_descriptor(gray_image, coords, num_bins=8, bin_size=4, angle_threshold=0.5):
    ret_coords = []
    ret_descriptors = []
    
    max_height = gray_image.shape[0]
    max_width = gray_image.shape[1]
    
    for xcord, ycord in coords:
        if xcord-8 < 0 or xcord+8 >= max_height:
            continue
        if ycord-8 < 0 or ycord+8 >= max_width:
            continue
    
        patch = gray_image[xcord-8:xcord+8, ycord-8:ycord+8]
        
        dx = cv2.Sobel(patch, ddepth=cv2.CV_64F, dx=1, dy=0)
        dy = cv2.Sobel(patch, ddepth=cv2.CV_64F, dx=0, dy=1)

        orientation = np.arctan2(dy, dx) * 180 / np.pi

        sub_blocks = [(i, j) for i in range(0, 16, bin_size) for j in range(0, 16, bin_size)]

        descriptor = np.zeros((num_bins * bin_size ** 2,))

        for x, y in sub_blocks:
            histogram = np.zeros((num_bins,))
            for i in range(bin_size):
                for j in range(bin_size):
                    bin_index = np.int32(orientation[x+i, y+j] / (360 / num_bins))
                    histogram[bin_index] += 1
            histogram.sort()
            histogram[int(angle_threshold * num_bins):] = 0
            descriptor[x + (y * bin_size):x + y * bin_size + num_bins] = histogram[:]
        
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm
        
        ret_coords.append((xcord, ycord))
        ret_descriptors.append(descriptor)
    
    return np.array(ret_coords), np.array(ret_descriptors)

def get_matching(pos_image_1, desc_image_1, pos_image_2, desc_image_2, min_dist=1):
    paired_coords = []
    # iterating thru all the elements of image 1
    for coord, desc in zip(pos_image_1, desc_image_1):
        # calculating the distance of point coord to all the other points in image 2
        calculated_dist = np.linalg.norm(desc_image_2 - desc, axis=1)
        selected_index = np.argmin(calculated_dist)
        # if the minimum distance is less than minimum specified then delete element at the selected index
        if calculated_dist[selected_index] <= min_dist:
            paired_coords.append((coord, pos_image_2[selected_index]))
            pos_image_2 = np.delete(pos_image_2, selected_index, axis=0)
            desc_image_2 = np.delete(desc_image_2, selected_index, axis=0)
    
    return paired_coords

if __name__ == '__main__':
    root_path = './images'
    image_filepath_1 = 'calculator-1.jpg'
    image_filepath_2 = 'calculator-2.jpg'
    image_1 = cv2.imread(os.path.join(root_path, image_filepath_1))
    image_2 = cv2.imread(os.path.join(root_path, image_filepath_2))

    image_1 = cv2.resize(image_1, resize_tuple_scaled(image_1.shape, 8))
    image_2 = cv2.resize(image_2, resize_tuple_scaled(image_2.shape, 8))
    
    # getting gray images
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    
    # getting corner's coordinates
    corner_coords_image_1 = get_harris_corners_coords(gray_image_1, threshold=0.01)
    corner_coords_image_2 = get_harris_corners_coords(gray_image_2, threshold=0.01)

    # converting images to print on matplotlib
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

    # getting sift descriptors for each point gotten with harris corner detection
    coords_image_1, descriptors_image_1 = compute_sift_descriptor(gray_image_1, corner_coords_image_1)
    coords_image_2, descriptors_image_2 = compute_sift_descriptor(gray_image_2, corner_coords_image_2)

    # pairing coords that euclid distance is the same
    paired_coords = get_matching(coords_image_1, descriptors_image_1, coords_image_2, descriptors_image_2, 0.5)
    
    # drawing rectangles for each corner centroid
    to_draw_image_1 = np.copy(image_1)
    draw_all_rectangles(to_draw_image_1, corner_coords_image_1)

    to_draw_image_2 = np.copy(image_2)
    draw_all_rectangles(to_draw_image_2, corner_coords_image_2)

    # getting dimensions of first image to correct second coordinates of parited coords
    width_image_1 = to_draw_image_1.shape[1]

    # creating single image to show with matches paired with a line
    show_image = concat_imgs(to_draw_image_1, to_draw_image_2)
    c_index = 0
    for coord in paired_coords:
        coord[1][1] += width_image_1
        show_image[skimage.draw.line(*coord[0], *coord[1])] = [255, 197, 255]

    # plottig images
    fig, axes = plt.subplots(1, 1, figsize=(15, 15))
    axes.imshow(show_image)
    axes.set_aspect('auto')
    
    plt.show()