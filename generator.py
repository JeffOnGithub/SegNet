# -*- coding: utf-8 -*-
import numpy as np
import cv2
from random import randint

# generator that we will use to read the data from the directory
def data_generator(img_dir, mask_dir, lists, batch_size, dims, n_labels, crop, flip):
        while True:
            yield single_batch_generator(img_dir, mask_dir, lists, batch_size, dims, n_labels, crop, flip)

# Generate one batch of data
def single_batch_generator(img_dir, mask_dir, lists, batch_size, dims, n_labels, crop, flip):
    ix = np.random.choice(np.arange(len(lists)), batch_size)
    imgs = []
    labels = []
    for i in ix:
        
        # images
        original_img = cv2.imread(img_dir + lists.iloc[i, 0] + ".png")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        # masks
        original_mask = cv2.imread(mask_dir + lists.iloc[i, 0] + '.png')
        
        # Crop, resize, flip
        transformed_img, transformed_mask = transform_data(original_img, original_mask, dims, crop, flip)
        
        # Convert mask to labels
        array_mask = to_categorical_labels(transformed_mask[:, :, 0], dims, n_labels)
        
        # Append image and mask to main lists
        imgs.append(transformed_img)
        labels.append(array_mask)
        
    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels

# Reimplementation of to_categorical, not sure if it will stay
def to_categorical_labels(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]]=1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x

# Geometric transformations of images and mask
def transform_data(original_img, original_mask, dims, crop, flip):
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
    