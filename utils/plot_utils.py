import os, signal
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2



def plot_correspondences(im1, im2, points1, points2, title1='From image 1', title2='To image 2', filepath=None):
    color_str = "orange"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.imshow(im1, cmap='gray', aspect='equal')
    ax1.set_title(title1)
    ax2.imshow(im2, cmap='gray',aspect='equal')
    ax2.set_title(title2)
    for i in range(len(points1)):
        ax1.plot([points1[i,0]], [points1[i,1]], marker='*', markersize=15, color=color_str)
        ax1.text(points1[i,0]+5, points1[i,1]-10, str(i), fontsize=15, color=color_str)
        ax2.plot([points2[i,0]], [points2[i,1]], marker='*', markersize=15, color=color_str)
        ax2.text(points2[i,0]+5, points2[i,1]-10, str(i), fontsize=15, color=color_str)
    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()
    plt.draw()
    if filepath:
        fig.savefig(filepath)
    # plt.show()

def show_image_pair(im1, im2, title1='Image 1', title2='Image 2', filepath=None):
    color_str = "orange"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.imshow(im1, cmap='gray', aspect='equal')
    ax1.set_title(title1)
    ax2.imshow(im2, cmap='gray', aspect='equal')
    ax2.set_title(title2)
    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()
    plt.draw()
    if filepath:
        fig.savefig(filepath)
