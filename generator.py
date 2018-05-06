# -*- coding: utf-8 -*-
"""Generator module for Segnet"""

from random import randint, uniform
from keras.utils import to_categorical
import numpy as np
import cv2

def domain_generator(img_dir,
                     domain_dir,
                     img_list,
                     domain_list,
                     batch_size,
                     dims,
                     crop,
                     flip,
                     motion_blur,
                     sp_noise):
    """Continous generator"""
    while True:
        imgs = []
        labels = []
        
        for i in range(0, batch_size):
            img_path = ''
            if randint(0, 1):
                # training image
                img_path = img_dir + img_list.iloc[1, 0]
                labels.append(0)
            else:
                # domain adaptation image
                img_path = domain_dir + domain_list.iloc[1, 0]
                labels.append(1)
            
            original_img = cv2.imread(img_path)
            # switch colors to RGB
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Geometric transformations to fit the network
            transformed_img, transformed_mask = transform_data(original_img, original_img, dims, crop, flip)
            
            # Data augmentations
            if uniform(0, 1) < motion_blur:
                transformed_img = motion_blur_image(transformed_img)
            
            if uniform(0, 1) < sp_noise:
                transformed_img = sp_noise_image(transformed_img)
            
            # Append image to main list
            imgs.append(transformed_img)
            
        yield np.array(imgs), to_categorical(labels, 2)
        
def segnet_generator(img_dir,
                     mask_dir,
                     lists,
                     batch_size,
                     dims,
                     n_labels,
                     crop,
                     flip,
                     motion_blur,
                     sp_noise):
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

def single_batch_generator(img_dir,
                           mask_dir,
                           lists,
                           batch_size,
                           dims,
                           n_labels,
                           crop, 
                           flip,
                           motion_blur=0,
                           sp_noise=0,
                           empty_mask=False):
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
        if empty_mask:
            array_mask = np.zeros([dims[0], dims[1], n_labels])    
        else:
            array_mask = to_categorical(transformed_mask[:, :, 0], n_labels)
            array_mask = array_mask.reshape(dims[0] * dims[1], n_labels)
        
        # Append image and mask to main lists
        imgs.append(transformed_img)
        labels.append(array_mask)
        
    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels

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
        orientation = randint(0, 3)
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

def sp_noise_image(image, salt_value=40):

    noise = np.random.randint(salt_value + 1, size = (image.shape[0], image.shape[1]))
    
    #---------- Pepper ----------#
    index = np.where(noise == 0)
    A = index[0]
    B = index[1]
    image[A,B,:] = 0
    
    #---------- Salt ----------#
    index = np.where(noise == salt_value)
    A = index[0]
    B = index[1]
    image[A,B,:] = 255
    
    return image