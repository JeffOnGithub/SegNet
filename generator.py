# -*- coding: utf-8 -*-
import numpy as np
import cv2
from random import randint

def to_categorical_labels(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]]=1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x

# generator that we will use to read the data from the directory
def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels, crop, flip):
        while True:
            ix = np.random.choice(np.arange(len(lists)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                
                # images
                original_img = cv2.imread(img_dir + lists.iloc[i, 0]+".png")
                # masks
                original_mask = cv2.imread(mask_dir + lists.iloc[i, 0] + '.png')
                
                # Random crop or resize, openCV starts with height, not width
                random_x = randint(0, original_img.shape[1] - dims[0])
                random_y = randint(0, original_img.shape[0] - dims[1])
                
                if crop:
                    transformed_img = original_img[random_y:random_y + dims[1], random_x:random_x + dims[0]]
                    transformed__mask = original_mask[random_y:random_y + dims[1], random_x:random_x + dims[0]]
                else:
                    transformed_img = cv2.resize(original_img, (dims[0], dims[1]))
                    transformed__mask = cv2.resize(original_mask, (dims[0], dims[1]))
               
                #Flip randomly images and masks
                if flip:
                    orientation = randint(0, 4)
                    if orientation == 0: #horizontal
                        transformed_img = cv2.flip(transformed_img, 0)
                        transformed__mask = cv2.flip(transformed__mask, 0)
                    elif orientation == 1: #vertical
                        transformed_img = cv2.flip(transformed_img, 1)
                        transformed__mask = cv2.flip(transformed__mask, 1)
                    elif orientation == 2: #horizontal and vertical
                        transformed_img = cv2.flip(transformed_img, -1)
                        transformed__mask = cv2.flip(transformed__mask, -1)
                    elif orientation == 3: #none
                        pass
                
                # Convert mask to labels
                array_mask = to_categorical_labels(transformed__mask[:, :, 0], dims, n_labels)
                
                # Append image and mask to main lists
                imgs.append(transformed_img)
                labels.append(array_mask)
                
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels
