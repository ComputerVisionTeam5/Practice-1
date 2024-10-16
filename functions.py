import matplotlib.pyplot as plt
import cv2 as cv
import time
from IPython.display import Image
import numpy as np
import random
import pandas as pd

def plot_img(n, figsize,titles,imgs, n_row=1):
    """
    Plots multiple images in a single row with specified titles.

    Parameters:
    - n (int): Number of images to plot.
    - figsize (tuple): Size of the figure (width, height).
    - titles (list of str): List of titles for the images.
    - imgs (list of numpy arrays): List of images to be plotted. Images should be in BGR format.
    """
    x, y = figsize
    fig, axes = plt.subplots(n_row, n // n_row, figsize=(x, y))
    axes = axes.ravel()
    for i in range(n):
        axes[i].imshow(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB))
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def homomorphic_filter(img, d0=30, gammaL=0.5, gammaH=1.5):
    """
    Applies a homomorphic filter to enhance the image by reducing uneven lighting.
    
    Parameters:
    - img: Input binary image.
    - d0: Cutoff frequency for the filter.
    - gammaL: Lower frequency gain (illumination component).
    - gammaH: Higher frequency gain (reflectance component).
    
    Returns:
    - img_filtered: The filtered image.
    """
    # Convert image to float and apply logarithm
    img_log = np.log1p(np.array(img, dtype="float"))

    # Perform Fourier Transform
    img_fft = np.fft.fft2(img_log)
    img_fft_shift = np.fft.fftshift(img_fft)

    # Create a high-pass filter
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  # Center of the image
    mask = np.ones((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = (gammaH - gammaL) * (1 - np.exp(- (dist ** 2) / (2 * (d0 ** 2)))) + gammaL

    # Apply the high-pass filter to the FFT of the image
    img_fft_filtered = img_fft_shift * mask

    # Inverse FFT to convert back to the spatial domain
    img_ifft_shift = np.fft.ifftshift(img_fft_filtered)
    img_ifft = np.fft.ifft2(img_ifft_shift)
    img_filtered = np.exp(np.real(img_ifft)) - 1

    # Normalize the result to the range [0, 255] and convert to uint8
    img_filtered = np.uint8(cv.normalize(img_filtered, None, 0, 255, cv.NORM_MINMAX))

    #plot_img(2, (6, 3), ["Before", "After"], [img, img_filtered])
            
    return img_filtered

def crop_image(img, cm = 1, pixels_per_cm = 37):
    minus = cm * pixels_per_cm
    height, width = img.shape[:2]
    return(img[minus:height-minus, minus:width-minus])

def pipeline_features(color_img, ID):
    color_img = crop_image(color_img)
    copy_img = color_img.copy()

    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    homo_img = homomorphic_filter(gray_img)

    _, binary_img = cv.threshold(homo_img, 60, 255, cv.THRESH_BINARY)

    binary_img = -binary_img + 255

    kernel = np.ones((5,5),np.uint8)

    dilated_img = cv.dilate(binary_img,kernel,iterations = 2)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(dilated_img)
    
    label_hue = np.uint8(179 * labels / np.max(labels)) 
    blank_ch = 255 * np.ones_like(label_hue)

    labeled_image = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)  
    labeled_image[label_hue == 0] = 0

    rows = []

    for i in range(1, num_labels):
        component_mask = np.uint8(labels == i) * 255  # Binary mask for the current component

        contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        perimeter = cv.arcLength(contours[0], True)
        area = cv.contourArea(contours[0])
        moments = cv.moments(contours[0])
        hu_moments = cv.HuMoments(moments).flatten()

        num_holes = stats[i, cv.CC_STAT_AREA] - area  # Approximating the number of holes
        euler_number = 1 - num_holes  # Euler number formula

        row = [ID, euler_number, perimeter, area] + hu_moments.tolist()

        rows.append(row)

    columns = ['ID', 'EulerNumber', 'Perimeter', 'Area', 'Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7']
    df = pd.DataFrame(rows, columns=columns)
    df = df[(df['Perimeter'] > 10)] # removing small dots
    upper_bound = df['Perimeter'].mean() + 350
    lower_bound = df['Perimeter'].mean() - 350
    df_filtered = df[(df['Perimeter'] >= lower_bound) & (df['Perimeter'] <= upper_bound)]

    selected_labels = df_filtered.index + 1  # Index starts at 0, label starts at 1

    for i in range(1, num_labels):
        if i in selected_labels:  # Only color selected components
            component_mask = np.uint8(labels == i) * 255  # Binary mask for the current component
            contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(copy_img, contours, -1, (0, 255, 0), 2)  # Color the selected components

    plot_img(3, (9, 4.5), ["Color image (contoured)", "Binary", "Connected"], [copy_img, dilated_img, labeled_image])

    return df_filtered