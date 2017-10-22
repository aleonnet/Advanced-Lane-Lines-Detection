import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import matplotlib.image as mpimg
from PIL import Image


# helper to display images in a path
def peek_img_path(path, imgs_per_line=4, w=16, h=20):
    imgs = []
    for file in os.listdir(path):
        imgs.append(Image.open(path + file))

    num = len(imgs)

    plt.figure(figsize=(w, h))
    if num <= imgs_per_line:
        for i in range(num):
            plt.subplot(1, num, i + 1)
            plt.imshow(imgs[i])
    else:
        i = 0
        for j in range(imgs_per_line):
            while i < num:
                plt.subplot(num, imgs_per_line, i + 1)
                plt.imshow(imgs[i])
                i += 1


# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255), chan='s'):
    chan_map = {'h': 0, 'l': 1, 's': 2}
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    chan_select = hls[:, :, chan_map[chan]]

    binary_output = np.zeros_like(chan_select)
    binary_output[(chan_select >= thresh[0]) & (chan_select <= thresh[1])] = 1

    return binary_output


def read_imgs_from_path(path):
    imgs = []
    for file in os.listdir(path):
        imgs.append(mpimg.imread(path + file))
    return imgs


def peek_imgs_list(imgs_list, imgs_per_line=4, w=16, h=20, title='title'):
    num = len(imgs_list)

    plt.figure(figsize=(w, h))
    if num <= imgs_per_line:
        for i in range(num):
            plt.subplot(1, num, i + 1)
            plt.imshow(imgs_list[i])
            plt.title(title)
    else:
        i = 0
        for j in range(imgs_per_line):
            while i < num:
                plt.subplot(num, imgs_per_line, i + 1)
                plt.imshow(imgs_list[i])
                plt.title(title)
                i += 1


def img_select(img, chan='s'):
    chan_map = {'h': 0, 'l': 1, 's': 2}
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    chan_select = hls[:, :, chan_map[chan]]

    # I want higher pixel value corresponds to a lighter saturation (lane lines)
    gray = np.dstack((chan_select, chan_select, chan_select))

    return gray


# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def img_select(img, grad='ab', sobel_kernel=3, orient='x', thresh=(0, 255)):
    if grad == 'ab':
        img = abs_sobel_thresh(img, sobel_kernel, orient, thresh)

    if grad == 'mag':
        img = mag_thresh(img, sobel_kernel, thresh)

    if grad == 'dir':
        img = dir_threshold(img, sobel_kernel, thresh)

    return img


def get_lane_curvature_off_center_distance(img_shape, curvature_tangent_pos_row, left_win_pos_col, left_win_pos_row,
                                           right_win_pos_col, right_win_pos_row,
                                           left_col_base, right_col_base, m_col_pix=3.7 / 700, m_row_pix=30 / 720):
    # Curvature
    # Fit new polynomials to x,y in world space
    left_win_pos_row = [i * m_row_pix for i in left_win_pos_row]
    left_win_pos_col = [i * m_col_pix for i in left_win_pos_col]
    right_win_pos_row = [i * m_row_pix for i in right_win_pos_row]
    right_win_pos_col = [i * m_col_pix for i in right_win_pos_col]

    left_fit_world = np.polyfit(left_win_pos_row, left_win_pos_col, 2)
    right_fit_world = np.polyfit(right_win_pos_row, right_win_pos_col, 2)

    curvature_left = ((1 + (2 * left_fit_world[0] * curvature_tangent_pos_row + left_fit_world[1]) ** 2) ** 1.5) / (
        2 * left_fit_world[0])
    curvature_right = ((1 + (2 * right_fit_world[0] * curvature_tangent_pos_row + right_fit_world[1]) ** 2) ** 1.5) / (
        2 * right_fit_world[0])
    curvature = (curvature_left + curvature_right) / 2.0

    # Off-center distance
    lane_center = (left_col_base + right_col_base) // 2
    heading_center = img_shape[1] // 2
    off_center = ((lane_center - heading_center) / 2.0) * m_col_pix

    return {'curvature': curvature, 'off_center_distance': off_center}


def generate_lane_lines(binary_img_top_view, num_win=9, margin=100, num_pix_thresh=50, m_col_pix=3.7 / 700,
                        m_row_pix=30 / 720):
    """Generating lane lines from scratch
       using slide windows method
    """
    img = cv2.cvtColor(binary_img_top_view, cv2.COLOR_RGB2GRAY)
    shape = img.shape

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    left_col_base = np.argmax(histogram[:midpoint])
    right_col_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    h_win = np.int(img.shape[0] / num_win)
    # Identify the x (column) and y (row) positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzero_row = np.array(nonzero[0])
    nonzero_col = np.array(nonzero[1])

    # Current positions to be updated for each window
    left_col_current = left_col_base
    right_col_current = right_col_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_idxs = []
    right_lane_idxs = []

    # windows position: the center of each window
    left_win_pos_col = []
    left_win_pos_row = []
    right_win_pos_col = []
    right_win_pos_row = []
    left_win_pos = []
    right_win_pos = []

    # Step through the windows one by one
    for window in range(num_win):
        # Identify window boundaries in x (col) and y (row)
        win_row_low = img.shape[0] - (window + 1) * h_win
        win_row_high = img.shape[0] - window * h_win
        win_col_left_low = left_col_current - margin
        win_col_left_high = left_col_current + margin
        win_col_right_low = right_col_current - margin
        win_col_right_high = right_col_current + margin

        # Draw the windows on the output image
        # cv2.rectangle(out_img, (win_col_left_low, win_row_low), (win_col_left_high, win_row_high), (0, 255, 0), 2)
        # cv2.circle(out_img, (left_col_current, (win_row_low + win_row_high)//2), 50, (0,255,0), thickness=2, lineType=8)
        left_win_pos.append((left_col_current, (win_row_low + win_row_high) // 2))

        # cv2.rectangle(out_img, (win_col_right_low, win_row_low), (win_col_right_high, win_row_high), (0, 255, 0), 2)
        # cv2.circle(out_img, (right_col_current, (win_row_low + win_row_high)//2), 50, (0,255,0), thickness=2, lineType=8)
        right_win_pos.append((right_col_current, (win_row_low + win_row_high) // 2))

        left_win_pos_col.append(left_col_current)
        left_win_pos_row.append((win_row_low + win_row_high) // 2)
        right_win_pos_col.append(right_col_current)
        right_win_pos_row.append((win_row_low + win_row_high) // 2)

        # Identify the nonzero pixels in x and y within the window
        fallin_left_idxs = ((nonzero_row >= win_row_low) & (nonzero_row < win_row_high) &
                            (nonzero_col >= win_col_left_low) & (nonzero_col < win_col_left_high)).nonzero()[0]
        fallin_right_inds = ((nonzero_row >= win_row_low) & (nonzero_row < win_row_high) &
                             (nonzero_col >= win_col_right_low) & (nonzero_col < win_col_right_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_idxs.append(fallin_left_idxs)
        right_lane_idxs.append(fallin_right_inds)
        # If you found > num_pix_thresh pixels, recenter next window on their mean position
        if len(fallin_left_idxs) > num_pix_thresh:
            left_col_current = np.int(np.mean(nonzero_col[fallin_left_idxs]))
        if len(fallin_right_inds) > num_pix_thresh:
            right_col_current = np.int(np.mean(nonzero_col[fallin_right_inds]))

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_win_pos_row, left_win_pos_col, 2)
    right_fit = np.polyfit(right_win_pos_row, right_win_pos_col, 2)

    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)

    out_img[nonzero_row[left_lane_idxs], nonzero_col[left_lane_idxs]] = [255, 0, 0]
    out_img[nonzero_row[right_lane_idxs], nonzero_col[right_lane_idxs]] = [0, 0, 255]

    # Curvature and off-center distance
    curvature_off_center = get_lane_curvature_off_center_distance(shape, left_win_pos_row[0], left_win_pos_col,
                                                                  left_win_pos_row,
                                                                  right_win_pos_col, right_win_pos_row, left_col_base,
                                                                  right_col_base)

    return {'img': out_img,
            'left_fit': left_fit, 'right_fit': right_fit,
            'curvature': curvature_off_center['curvature'],
            'off_center_distance': curvature_off_center['off_center_distance'],
            'left_points': left_win_pos, 'right_points': right_win_pos,
            'left_points_col': left_win_pos_col, 'left_points_row': left_win_pos_row,
            'right_points_col': right_win_pos_col, 'right_points_row': right_win_pos_row}


def update_lane_lines(binary_img_top_view, left_fit_prev, right_fit_prev, margin=100):
    """
    Updating fit parameters based on previous fit
    """
    nonzero = binary_img_top_view.nonzero()
    nonzero_row = np.array(nonzero[0])
    nonzero_col = np.array(nonzero[1])

    left_lane_idxs = ((nonzero_col > (
        left_fit_prev[0] * (nonzero_row ** 2) + left_fit_prev[1] * nonzero_row + left_fit_prev[2] - margin)) & (
                          nonzero_col < (
                              left_fit_prev[0] * (nonzero_row ** 2) + left_fit_prev[1] * nonzero_row + left_fit_prev[
                                  2] + margin)))
    right_lane_idxs = ((nonzero_col > (
        right_fit_prev[0] * (nonzero_row ** 2) + right_fit_prev[1] * nonzero_row + right_fit_prev[2] - margin)) & (
                           nonzero_col < (
                               right_fit_prev[0] * (nonzero_row ** 2) + right_fit_prev[1] * nonzero_row +
                               right_fit_prev[
                                   2] + margin)))

    # Again, extract left and right line pixel positions
    left_col = nonzero_col[left_lane_idxs]
    left_row = nonzero_row[left_lane_idxs]
    right_col = nonzero_col[right_lane_idxs]
    right_row = nonzero_row[right_lane_idxs]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_row, left_col, 2)
    right_fit = np.polyfit(right_row, right_col, 2)

    # Curvature and off-center distance
    curvature_off_center = get_lane_curvature_off_center_distance(binary_img_top_view.shape, left_row[0], left_col,
                                                                  left_row,
                                                                  right_col, right_row, left_col[0], right_col[0])

    return {'left_fit': left_fit, 'right_fit': right_fit,
            'curvature': curvature_off_center['curvature'],
            'off_center_distance': curvature_off_center['off_center_distance']}


def mask(img, keep=(250, 1000)):
    shape = img.shape
    # keep_region = np.array([[(0,keep[0]), (shape[0],keep[0]), (shape[0],keep[1]), (0,keep[1])]])
    keep_region = np.array([[(keep[0], 0), (keep[0], shape[0]), (keep[1], shape[0]), (keep[1], 0)]])

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, keep_region, (255, 255, 255))

    return cv2.bitwise_and(img, mask)


# threshold values are tuned by trial-and-error method.
def comb_select(img, s_thresh=(128, 255), l_thresh=(64, 255), g_thresh=(30, 255), sobel_kernel=5):
    # create color thresholding mask
    chan_map = {'h': 0, 'l': 1, 's': 2}
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    chan_sel_s = hls[:, :, chan_map['s']]

    s_bin = np.zeros_like(chan_sel_s)
    s_bin[(chan_sel_s >= s_thresh[0]) & (chan_sel_s <= s_thresh[1])] = 1

    chan_sel_l = hls[:, :, chan_map['l']]

    l_bin = np.zeros_like(chan_sel_l)
    l_bin[(chan_sel_l >= s_thresh[0]) & (chan_sel_l <= s_thresh[1])] = 1

    # print("max: ", np.amax(chan_select), "min: ", np.amin(chan_select), "mean: ", np.mean(chan_select))
    # plt.figure()
    # plt.imshow(chan_select)

    # Rescale back to 8 bit integer
    # print("before", np.amin(chan_select), np.amax(chan_select))
    # chan_select = np.uint8(255*chan_select/np.max(chan_select))
    # print("after", np.amin(chan_select), np.amax(chan_select))

    # plt.figure()
    # plt.imshow(col_bin)

    # create gradient thresholding mask
    abs_sobel_x = np.absolute(cv2.Sobel(chan_sel_l, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    # print("abs max: ", np.amax(abs_sobel), "min: ", np.amin(abs_sobel), "mean: ", np.mean(abs_sobel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    # print("scaled max: ", np.amax(scaled_sobel), "min: ", np.amin(scaled_sobel), "mean: ", np.mean(scaled_sobel))
    # plt.figure()
    # plt.imshow(scaled_sobel)

    sx_bin = np.zeros_like(scaled_sobel)
    sx_bin[(scaled_sobel >= g_thresh[0]) & (scaled_sobel <= g_thresh[1])] = 1

    # combine masks
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[((s_bin == 1) & (l_bin == 1) | (sx_bin == 1))] = 1
    binary_output = 255 * np.dstack((binary_output, binary_output, binary_output)).astype('uint8')

    return binary_output


# assuming road is flat
# In reality, roads typically have different attitude orientations
# We can use gyroscope readings to correct this
# For this project, I will just assume the road is flat, which should be
# also a first-order approximation for reality

def persp_to_top_view(img,
                      src=np.float32([(180, 700),
                                      (565, 464),
                                      (735, 464),
                                      (1200, 700)]),
                      dst=np.float32([(300, 700),
                                      (300, 200),
                                      (1280 - 300, 200),
                                      (1280 - 300, 700)])):
    h, w = img.shape[:2]
    dst

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    top = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return M, Minv, top


def top_to_persp_view(img, Minv, w=1280, h=720):
    return cv2.warpPerspective(img, Minv, (w, h), flags=cv2.INTER_LINEAR)


corners = np.float32([(180, 700), (565, 464), (735, 464), (1200, 700)])

# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255), chan='s'):
    chan_map = {'h': 0, 'l': 1, 's': 2}
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    chan_select = hls[:, :, chan_map[chan]]

    binary_output = np.zeros_like(chan_select)
    binary_output[(chan_select >= thresh[0]) & (chan_select <= thresh[1])] = 1

    return binary_output


def read_imgs_from_path(path):
    imgs = []
    for file in os.listdir(path):
        imgs.append(mpimg.imread(path + file))
    return imgs


def peek_imgs_list(imgs_list, imgs_per_line=4, w=16, h=20, title='title'):
    num = len(imgs_list)

    plt.figure(figsize=(w, h))
    if num <= imgs_per_line:
        for i in range(num):
            plt.subplot(1, num, i + 1)
            plt.imshow(imgs_list[i])
            plt.title(title)
    else:
        i = 0
        for j in range(imgs_per_line):
            while i < num:
                plt.subplot(num, imgs_per_line, i + 1)
                plt.imshow(imgs_list[i])
                plt.title(title)
                i += 1


def img_select(img, chan='s'):
    chan_map = {'h': 0, 'l': 1, 's': 2}
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    chan_select = hls[:, :, chan_map[chan]]

    # I want higher pixel value corresponds to a lighter saturation (lane lines)
    gray = np.dstack((chan_select, chan_select, chan_select))

    return gray
