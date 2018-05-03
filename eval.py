# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import argparse
from SegNet import CreateSegNet
from generator import single_batch_generator

def main(args):
    pred_imgs, truth_maps = single_batch_generator(args.testimg_dir, 
                                                  args.testmsk_dir, 
                                                  pd.read_csv(args.test_list,header=None, dtype={0: str}), 
                                                  args.batch_size, 
                                                  [args.input_shape[0], args.input_shape[1]], 
                                                  args.n_labels, 
                                                  args.crop, 
                                                  args.flip)
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
    parser.add_argument("--batch_size",
            default=20,
            type=int,
            help="Eval batch size")
    parser.add_argument("--n_labels",
            default=2,
            type=int,
            help="Number of label")
    parser.add_argument("--crop",
            default=True,
            help="Crop to input shape, otherwise resize")
    parser.add_argument("--flip",
            default=True,
            help="Random flip of training images")
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
