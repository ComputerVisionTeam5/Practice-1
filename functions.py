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


def pipeline_features(color_img):
    copy_img = color_img.copy()
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    _, binary_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
    binary_img = -binary_img+255
    num_labels, labels = cv.connectedComponents(binary_img)

    label_hue = np.uint8(179 * labels / np.max(labels)) 
    blank_ch = 255 * np.ones_like(label_hue)

    labeled_image = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)  
    labeled_image[label_hue == 0] = 0
    
    for i in range(1, num_labels): 
        component_mask = np.uint8(labels == i) * 255  # Binary mask for the current component

        contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        perimeter = cv.arcLength(contours[0], True)
        area = cv.contourArea(contours[0])
        moments = cv.moments(contours[0])
        hu_moments = cv.HuMoments(moments).flatten()

        print(f"Component {i}:")
        print(f"  Perimeter: {perimeter:.2f} pixels")
        print(f"  Area: {area:.2f} pixels^2")
        print(f"  Hu Moments: {hu_moments}")

        masked_color_image = cv.bitwise_and(color_img, color_img, mask=component_mask)


        cv.drawContours(copy_img, contours, -1, (0, 255, 0), 2)

    plot_img(3, (9, 4.5), ["Color image (contoured)", "Binary", "Connected"], [copy_img, binary_img, labeled_image])


def pipeline_features(color_img, ID):
    copy_img = color_img.copy()

    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    _, binary_img = cv.threshold(gray_img, 150, 255, cv.THRESH_BINARY)

    binary_img = -binary_img + 255

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_img)
    
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

    plot_img(3, (9, 4.5), ["Color image (contoured)", "Binary", "Connected"], [copy_img, binary_img, labeled_image])

    return df_filtered