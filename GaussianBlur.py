import math
import numpy as np


def get_kernel(size):
    # sigma is square root of kernel size
    sigma = math.sqrt(size)

    # create empty kernel matrix
    kernel = np.zeros((size, size))

    # get values
    for row in range(size):
        for col in range(size):
            # get x and y distance from center where center is (0,0)
            x = row - size//2
            y = col - size//2

            # 2D gaussian function
            kernel[row, col] = (1 / (2 * np.pi * sigma ** 2)) * (np.e ** - ((x ** 2 + y ** 2) / 2 * sigma ** 2))

    return kernel


def gaussian_blur(img, kernel_size=5):
    # get kernel
    kernel = get_kernel(kernel_size)

    # make sure that kernel sum is equal to kernel matrix size
    # this will retain the color intensity of the resulting image
    kernel = kernel + (kernel_size ** 2 - kernel.sum())/kernel_size ** 2

    # reshape kernel to 3 channels
    kernel = np.repeat(kernel, 3).reshape((kernel_size, kernel_size, 3))

    # add padding to image
    pad_size = kernel_size//2
    padded = np.zeros((img.shape[0] + pad_size * 2, img.shape[1] + pad_size * 2, 3), dtype="uint8")
    padded[pad_size: img.shape[0] + pad_size, pad_size: img.shape[1] + pad_size] = img

    # create blank image for output
    conv_img = np.zeros(img.shape, dtype="uint8")

    # loop every pixel in the image
    for row in range(padded.shape[0]):
        for col in range(padded.shape[1]):
            if pad_size <= row < img.shape[0] + pad_size and pad_size <= col < img.shape[1] + pad_size:
                # get the start and end point of the matrix where the current pixel is the center
                x_start = row - pad_size
                y_start = col - pad_size

                x_end = row + pad_size + 1
                y_end = col + pad_size + 1

                # get the matrix from the padded image
                matrix = padded[x_start: x_end, y_start: y_end]

                # convolution
                # multiply the matrix with the kernel and then get the average
                conv_img[row-pad_size, col-pad_size] = np.mean(matrix * kernel, axis=0).mean(axis=0)

    return conv_img
