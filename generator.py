# -*- coding: utf-8 -*-
"""Generator module for Segnet"""

from random import randint, uniform, random
import numpy as np
import cv2

def data_generator(img_dir, mask_dir, lists, batch_size, dims, n_labels, crop, flip, motion_blur, sp_noise):
    """Continous generator"""
    while True:
        yield single_batch_generator(img_dir,
                                     mask_dir,
                                     lists,
                                     batch_size,
                                     dims,
                                     n_labels,
                                     crop,
                                     flip,
                                     motion_blur,
                                     sp_noise)

def single_batch_generator(img_dir, mask_dir, lists, batch_size, dims, n_labels, crop, flip, motion_blur=0, sp_noise=0, empty_mask=False):
    """Generate one batch of data"""
    ix = np.random.choice(np.arange(len(lists)), batch_size)
    imgs = []
    labels = []
    for i in ix:
        
        # images
        original_img = cv2.imread(img_dir + lists.iloc[i, 0])
        # switch colors to RGB
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        # masks
        original_mask = cv2.imread(mask_dir + lists.iloc[i, 0])
        
        # Geometric transformations to fit the network
        transformed_img, transformed_mask = transform_data(original_img, original_mask, dims, crop, flip)
        
        # Data augmentations
        if uniform(0, 1) < motion_blur:
            transformed_img = motion_blur_image(transformed_img)
        
        if uniform(0, 1) < sp_noise:
            transformed_img = sp_noise_image(transformed_img)
            
        # Convert mask to labels
        array_mask = to_categorical_labels(transformed_mask[:, :, 0], dims, n_labels, empty_mask)
        
        # Append image and mask to main lists
        imgs.append(transformed_img)
        labels.append(array_mask)
        
    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels

def to_categorical_labels(labels, dims, n_labels, empty_mask):
    """Works like to_categorical, but allow to returns empty masks"""
    x = np.zeros([dims[0], dims[1], n_labels])
    if not empty_mask:
        for i in range(dims[0]):
            for j in range(dims[1]):
                x[i, j, labels[i][j]] = 1
        x = x.reshape(dims[0] * dims[1], n_labels)
    return x

def transform_data(original_img, original_mask, dims, crop, flip):
    """Geometric transformations of images and mask"""
    # Random crop or resize, openCV starts with height, not width
    if crop:
        random_x = randint(0, original_img.shape[1] - dims[0])
        random_y = randint(0, original_img.shape[0] - dims[1])
        transformed_img = original_img[random_y:random_y + dims[1], random_x:random_x + dims[0]]
        transformed_mask = original_mask[random_y:random_y + dims[1], random_x:random_x + dims[0]]
    else:
        transformed_img = cv2.resize(original_img, (dims[0], dims[1]))
        transformed_mask = cv2.resize(original_mask, (dims[0], dims[1]))
   
    #Flip randomly images and masks
    if flip:
        orientation = randint(0, 4)
        if orientation == 0: #horizontal
            transformed_img = cv2.flip(transformed_img, 0)
            transformed_mask = cv2.flip(transformed_mask, 0)
        elif orientation == 1: #vertical
            transformed_img = cv2.flip(transformed_img, 1)
            transformed_mask = cv2.flip(transformed_mask, 1)
        elif orientation == 2: #horizontal and vertical
            transformed_img = cv2.flip(transformed_img, -1)
            transformed_mask = cv2.flip(transformed_mask, -1)
        elif orientation == 3: #none
            pass
    return transformed_img, transformed_mask

def motion_blur_image(image):
    """Degrade image quality - motion blur"""
    # Motion blur
    # generating the kernel
    orientation = randint(0, 4)
    kernel_motion_blur = np.zeros((3, 3))
    if orientation == 0: #horizontal
        kernel_motion_blur = np.array([[0, 0, 0],
                                       [1, 1, 1],
                                       [0, 0, 0]])
    elif orientation == 1: #vertical
        kernel_motion_blur = np.array([[0, 1, 0],
                                       [0, 1, 0],
                                       [0, 1, 0]])
    elif orientation == 2: #diagonal 1
        kernel_motion_blur = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
    elif orientation == 3: #diagonal 2
        kernel_motion_blur = np.array([[0, 0, 1],
                                       [0, 1, 0],
                                       [1, 0, 0]])
    
    kernel_motion_blur = kernel_motion_blur / 3
    
    # applying the kernel to the input image
    return cv2.filter2D(image, -1, kernel_motion_blur)

def sp_noise_image(image, prob=0.01):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output