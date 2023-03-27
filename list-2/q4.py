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
    
    lambda_max = harris_response.max()
    
    def neighbors_max(x, y):
        return harris_response[x-1:x+2, y-1:y+2].max()
    
    corners_coords = []
    for i, row in enumerate(harris_response[1:-2]):
        for j, pixel in enumerate(row[1:-2]):
            if pixel >= lambda_max * threshold and pixel >= neighbors_max(i+1, j+1):
                corners_coords.append((i+1, j+1))
    
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

def resize_tuple_scaled(shape, factor):
    return shape[1] // factor, shape[0] // factor

def rectangle_corners(center: tuple[int], diag: int) -> tuple[tuple[int, int]]:
    return (center[0] - diag, center[1] - diag), (center[0] + diag, center[1] + diag)

def draw_all_rectangles(image, coords):
    for coord in coords:
        try:
            image[skimage.draw.rectangle_perimeter(*rectangle_corners(coord, 2))] = [255, 0, 0]
        except:
            pass

def compute_sift_descriptor(gray_image, coords, num_bins=8, bin_size=4, threshold=0.8):
    assert threshold <= num_bins
    ret_descriptors = []
    
    max_height = gray_image.shape[0]
    max_width = gray_image.shape[1]
    
    sub_blocks = [(i, j) for i in range(0, 16, bin_size) for j in range(0, 16, bin_size)]
    
    max_freq = (bin_size ** 2) * threshold
    
    for xcord, ycord in coords:
        if xcord-8 < 0 or xcord+8 >= max_height:
            continue
        if ycord-8 < 0 or ycord+8 >= max_width:
            continue
    
        patch = gray_image[xcord-8:xcord+8, ycord-8:ycord+8]
        
        dx = cv2.Sobel(patch, ddepth=cv2.CV_64F, dx=1, dy=0)
        dy = cv2.Sobel(patch, ddepth=cv2.CV_64F, dx=0, dy=1)

        orientation = (np.arctan2(dy, dx) / np.pi) * 180.0

        descriptor = np.zeros((num_bins * bin_size ** 2,))

        for index, (x, y) in enumerate(sub_blocks):
            histogram = np.zeros((num_bins,))
            for i in range(bin_size):
                for j in range(bin_size):
                    bin_index = np.int32(orientation[x+i, y+j] / (360 / num_bins))
                    histogram[bin_index] += 1
            histogram[histogram > max_freq] = max_freq
            histogram.sort()
            descriptor[index * num_bins:index * num_bins + num_bins] = histogram[:]

        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm
        
        ret_descriptors.append(descriptor)
    
    return np.array(coords), np.array(ret_descriptors)

def get_matching(pos_image_1, desc_image_1, pos_image_2, desc_image_2, min_dist=1):
    paired_coords = []
    # iterating thru all the elements of image 1
    for coord, desc in zip(pos_image_1, desc_image_1):
        # calculating the distance of point coord to all the other points in image 2
        print(desc)
        calculated_dist = np.linalg.norm(desc_image_2 - desc, axis=1)
        selected_index = np.argmin(calculated_dist)
        # if the minimum distance is less than minimum specified then delete element at the selected index
        if calculated_dist[selected_index] <= min_dist:
            paired_coords.append((coord, pos_image_2[selected_index]))
            pos_image_2 = np.delete(pos_image_2, selected_index, axis=0)
            desc_image_2 = np.delete(desc_image_2, selected_index, axis=0)
        
        if len(desc_image_2) == 0:
            break
    
    return paired_coords

if __name__ == '__main__':
    root_path = './images'
    # image_filepath_1 = 'calculator-1.jpg'
    # image_filepath_2 = 'calculator-2.jpg'
    # image_filepath_1 = 'image_1.png'
    # image_filepath_2 = 'image_2.png'
    image_filepath_1 = 'cow_1.png'
    image_filepath_2 = 'cow_2.png'
    image_1 = cv2.imread(os.path.join(root_path, image_filepath_1))
    image_2 = cv2.imread(os.path.join(root_path, image_filepath_2))

    image_1 = cv2.resize(image_1, resize_tuple_scaled(image_1.shape, 2))
    image_2 = cv2.resize(image_2, resize_tuple_scaled(image_2.shape, 2))
    
    # getting gray images
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    
    # getting corner's coordinates
    if True: # using opencv's harris to calculate points
        def get_corners(gray_image, k=0.01):
            dst = cv2.cornerHarris(np.float32(gray_image), 2, 3, k)
            
            corners = np.argwhere(dst > 0.01 * dst.max())
            corners = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in corners]
            return corners

        corners_image_1 = get_corners(gray_image_1, 0.04)
        corners_image_2 = get_corners(gray_image_2, 0.04)
        
        
        
        # print(corners_image_1)
        
        # cv2.imshow('test_image_1', corners_image_1)
        # cv2.imshow('test_image_2', corners_image_2)
        # cv2.waitKey(0)
    if False: # using self-made harris to calulate points
        corners_image_1 = get_harris_corners_coords(gray_image_1, threshold=0.1)
        corners_image_2 = get_harris_corners_coords(gray_image_2, threshold=0.1)
    if False: # using opencv's sift to calculate points
        sift = cv2.SIFT_create(nfeatures=50)
        corners_image_1 = cv2.KeyPoint_convert(sift.detect(gray_image_1, None))
        corners_image_1 = np.array(corners_image_1, dtype=np.int32)
        corners_image_2 = cv2.KeyPoint_convert(sift.detect(gray_image_2, None))
        corners_image_2 = np.array(corners_image_2, dtype=np.int32)
        
        corners_image_1 = np.flip(corners_image_1, axis=None)
        corners_image_2 = np.flip(corners_image_2, axis=None)
    
    if False:
        kp = sift.detect(gray_image_1, None)

        pts = cv2.KeyPoint_convert(kp)
        
        for k in pts:
            print(k)
        
        to_show = np.copy(gray_image_1)
        test_img = cv2.drawKeypoints(to_show, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow('test_image', test_img)
        cv2.waitKey(0)
        
        # print(kp)
        
        exit()

    # converting images to print on matplotlib
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

    # getting sift descriptors for each point gotten with harris corner detection
    
    # corners_image_1 = np.array(corners_image_1)
    # corners_image_1 = np.flip(corners_image_1, axis=None)
    def get_converted_points(corners):
        final_list = []
        for corner in corners:
            pt = cv2.KeyPoint(corner[1], corner[0], 1)
            final_list.append(pt)
        return final_list

    # corners_image_1 = np.array(get_converted_points(corners_image_1))
    # corners_image_2 = np.array(get_converted_points(corners_image_2))
    

    # corners_image_1, descriptors_image_1 = sift.compute(gray_image_1, corners_image_1)
    # corners_image_2, descriptors_image_2 = sift.compute(gray_image_2, corners_image_2)

    if False:
        # Computing corners with self-made sift descriptor
        corners_image_1, descriptors_image_1 = compute_sift_descriptor(gray_image_1, corners_image_1, threshold=0.8)
        corners_image_2, descriptors_image_2 = compute_sift_descriptor(gray_image_2, corners_image_2, threshold=0.8)

        # pairing coords that euclid distance is the same
        paired_coords = get_matching(corners_image_1, descriptors_image_1, corners_image_2, descriptors_image_2, 100)
    
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)
    
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # matches = flann.knnMatch(descriptors_image_1, descriptors_image_2, k=2)
    
    nums_features = [5, 10, 20]
    contrast_thresholds = [0.03, 0.07, 0.1]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    for i, (num_features, contrast) in enumerate(zip(nums_features, contrast_thresholds)):
        sift = cv2.SIFT_create(nfeatures=num_features, contrastThreshold=contrast)
        corners_image_1, descriptors_image_1 = sift.detectAndCompute(gray_image_1, None)
        corners_image_2, descriptors_image_2 = sift.detectAndCompute(gray_image_2, None)
        
        brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        no_of_matches = brute_force.match(descriptors_image_1, descriptors_image_2)
        no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)

        show_image = cv2.drawMatches(image_1, corners_image_1, image_2, corners_image_2, no_of_matches, None, flags=2)
        
        axes[i].set_aspect('auto')
        axes[i].imshow(show_image)
        axes[i].set_title(f'fetures {num_features} contrast {contrast}')

    plt.show()

    # drawing rectangles for each corner centroid
    # to_draw_image_1 = np.copy(image_1)
    # draw_all_rectangles(to_draw_image_1, corners_image_1)

    # to_draw_image_2 = np.copy(image_2)
    # draw_all_rectangles(to_draw_image_2, corners_image_2)

    # getting dimensions of first image to correct second coordinates of parited coords
    # width_image_1 = to_draw_image_1.shape[1]

    # print(to_draw_image_1.shape)
    # print(to_draw_image_2.shape)

    # creating single image to show with matches paired with a line
    # show_image = concat_imgs(to_draw_image_1, to_draw_image_2)
    # c_index = 0
    # colors = np.array([[197, 0, 197], [0, 197, 0], [0, 0, 197], [197, 197, 0], [0, 197, 197]])
    # for coord in paired_coords:
    #     # print(*coord[0], *coord[1])
        
    #     coord[1][1] += width_image_1
    #     show_image[skimage.draw.line(*coord[0], *coord[1])] = colors[c_index]
    #     c_index += 1
    #     c_index %= len(colors)

    # matchesMask = [[0, 0] for i in range(len(matches))]
    # for i, (m, n) in enumerate(matches):
    #     if m.distance <= 0.7*n.distance:
    #         matchesMask[i] = [1, 0]

    # draw_params = dict(matchColor = (0,255,0),
    #                singlePointColor = (255,0,0),
    #                matchesMask = matchesMask,
    #                flags = cv2.DrawMatchesFlags_DEFAULT)

    # show_image = cv2.drawMatchesKnn(
    #     to_draw_image_1, corners_image_1,
    #     to_draw_image_2, corners_image_2,
    #     matches, None, **draw_params)

    # axes.imshow(show_image)
    # axes.set_aspect('auto')
    
    # plt.show()
    # plottig images
    # fig, axes 