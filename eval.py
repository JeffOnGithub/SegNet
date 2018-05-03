# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
from random import randint
import argparse
from SegNet import CreateSegNet

random_x = randint(0, 511)
random_y = randint(0, 255)

def main(args):
    truth_maps = []
    pred_imgs = []
    
    for i in range(0, args.sample_size):
        # Load ground truth maps
        full_path = args.testmsk_dir + str(i).zfill(5) + '.png'
        img = cv2.imread(full_path)
        cropped_img = img[random_y:random_y + 256, random_x:random_x + 256]
        truth_maps.append(cropped_img)
    
        # Load test images    
        full_path = args.testimg_dir + str(i).zfill(5) + '.png'
        img_map = cv2.imread(full_path)
        cropped_map = img_map[random_y:random_y + 256, random_x:random_x + 256]
        cropped_map = cv2.cvtColor(cropped_map, cv2.COLOR_RGB2BGR)
        pred_imgs.append(cropped_map)
    np_pred_imgs = np.array(pred_imgs)
    
    # Generate a combined images of all test images input
    imgs_comb = np.hstack( (np.asarray(i) for i in pred_imgs ) )
    #plt.imsave('combined_input.png', cv2.cvtColor(imgs_comb, cv2.COLOR_BGR2RGB))
    
    # Build a network and load weights
    segnet = CreateSegNet(args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode)
    print("Segnet built")
    segnet.load_weights(args.weights)
    print("Weights loaded")
    
    # Run images in the network
    result = segnet.predict(np_pred_imgs)
    print("Predictions generated")
    
    # Reshape result images
    result_imgs = []
    for image in result:
        result_img = []
        for i in range(0, args.n_labels):
            reshaped = np.reshape(image[:,i], (args.input_shape[0], args.input_shape[1]))
            result_img.append(reshaped)
        result_imgs.append(result_img)
    
    # Generate a combined images of all test images output
    v_stacked = []
    for image in result_imgs:
        this_image = np.vstack( np.asarray(i) for i in image)
        v_stacked.append(this_image)
    
    results_comb = np.hstack( (np.asarray(i) for i in v_stacked ) )
    #plt.imsave('combined_output.png', results_comb)
    
    # Stack combined images
    imgs_comb = imgs_comb / 255.0
    results_comb = cv2.cvtColor(results_comb, cv2.COLOR_GRAY2RGB)
    
    imgs_to_stack = [imgs_comb, results_comb]
    imgs_total = np.vstack( (np.asarray(i) for i in imgs_to_stack ) )
    
    # Save result
    plt.imsave(args.results_dir + 'combined.png', imgs_total)
    
    print("Results compilation saved")

if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet dataset")
    parser.add_argument("--weights",
            default="./weights/SegNet.hdf5",
            help="starting weights path")
    parser.add_argument("--model",
        default="./model/SegNet.json",
        help="starting weights path")
    parser.add_argument("--test_list",
            default="./dataset/test/id.txt",
            help="test list path")
    parser.add_argument("--testimg_dir",
            default="./dataset/test/images/",
            help="test image dir path")
    parser.add_argument("--testmsk_dir",
            default="./dataset/test/maps/",
            help="test mask dir path")
    parser.add_argument("--results_dir",
            default="./results/",
            help="test mask dir path")
    parser.add_argument("--sample_size",
            default=20,
            type=int,
            help="Eval sample size")
    parser.add_argument("--n_labels",
            default=2,
            type=int,
            help="Number of label")
    parser.add_argument("--input_shape",
            default=(256, 256, 3),
            help="Input images shape")
    parser.add_argument("--kernel",
            default=3,
            type=int,
            help="Kernel size")
    parser.add_argument("--pool_size",
            default=(2, 2),
            help="pooling and unpooling size")
    parser.add_argument("--output_mode",
            default="softmax",
            type=str,
            help="output activation")
    args = parser.parse_args()
    main(args)
