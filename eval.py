# -*- coding: utf-8 -*-
"""Evaluate the Segnet model"""

import argparse
from os import listdir, environ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from segnet import create_segnet
from generator import single_batch_generator
from configuration import CONFIG

#Set CUDA device for tensorflow
environ["CUDA_VISIBLE_DEVICES"] = CONFIG['segnet']['cuda_device']

def compare_image_ground_truth(compared_image, ground_truth):
    """Compare a prediction to ground truth to establish a visual result of the accuracy"""
    #Start with an empty array
    diff = np.zeros([compared_image.shape[0], compared_image.shape[1], 3])
    #Loop everything
    for x in range(0, compared_image.shape[0]):
        for y in range(0, compared_image.shape[1]):
            image_data = compared_image[x][y]
            truth_data = ground_truth[x][y]
            #Yellow to Green color scale for zones where it should be 1
            if truth_data == 1:
                delta = truth_data - image_data
                diff[x, y] = [delta, 1, 0]
            #Red to black color scale for zones where it should be 0
            else:
                delta = image_data
                diff[x, y] = [delta, 0, 0]
    return diff

def main(args):
    """Evaluate model"""
    pred_imgs, truth_masks = single_batch_generator(args.testimg_dir,
                                                    args.testmsk_dir,
                                                    pd.DataFrame(listdir(args.testimg_dir)),
                                                    args.batch_size,
                                                    [args.input_shape[0], args.input_shape[1]],
                                                    args.n_labels,
                                                    args.crop,
                                                    args.flip,
                                                    empty_mask=not args.ground_truth)
                                                    #do not create masks when they are not needed, useful for tests
                                                    #without mask data
    np_pred_imgs = np.array(pred_imgs)

    # Generate a combined images of all test images input
    imgs_comb = np.hstack((np.asarray(i) for i in pred_imgs))
    #plt.imsave('combined_input.png', cv2.cvtColor(imgs_comb, cv2.COLOR_BGR2RGB))

    # Build a network and load weights
    segnet, domain = create_segnet(args.input_shape,
                           args.n_labels,
                           args.kernel,
                           args.pool_size,
                           args.output_mode)
    print("Segnet built")
    segnet.load_weights(args.weights)
    print("Weights loaded")
    
    # Run images in the network and get predictions
    result = segnet.predict(np_pred_imgs)
    print("Predictions generated")
        
    # Reshape result images to their initial shape
    result_imgs = []
    for image in result:
        result_img = []
        for i in range(0, args.n_labels):
            reshaped = np.reshape(image[:, i], (args.input_shape[0], args.input_shape[1]))
            result_img.append(reshaped)
        result_imgs.append(result_img)
    
    # Generate a combined images of all test images output
    v_stacked = []
    for image in result_imgs:
        this_image = np.vstack(np.asarray(i) for i in image)
        v_stacked.append(this_image)

    results_comb = np.hstack((np.asarray(i) for i in v_stacked))
    #plt.imsave('combined_output.png', results_comb)
    
    #Compare to ground truth if selected, otherwise we will output raw results
    if args.ground_truth:
        # Reshape ground truth to compare to predictions
        truth_maps_reshaped = []
        for mask in truth_masks:
            result_mask = []
            for i in range(0, args.n_labels):
                reshaped = np.reshape(mask[:, i], (args.input_shape[0], args.input_shape[1]))
                result_mask.append(reshaped)
            truth_maps_reshaped.append(result_mask)
            
        # Generate a combined images of all ground truth masks
        v_stacked = []
        for mask in truth_maps_reshaped:
            this_mask = np.vstack(np.asarray(i) for i in mask)
            v_stacked.append(this_mask)
        
        masks_comb = np.hstack((np.asarray(i) for i in v_stacked))
        #plt.imsave('combined_output.png', masks_comb)
        
        #Compared ground truth to predictions
        results_comb = compare_image_ground_truth(results_comb, masks_comb)
    
    else:    
        results_comb = cv2.cvtColor(results_comb, cv2.COLOR_GRAY2RGB)
        
    # Stack combined images
    imgs_comb = imgs_comb / 255.0
    imgs_to_stack = [imgs_comb, results_comb]
    imgs_total = np.vstack((np.asarray(i) for i in imgs_to_stack))
    
    # Save compilation result
    plt.imsave(args.results_dir + 'combined.png', imgs_total)
    
    print("Results compilation saved")

if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet dataset")
    parser.add_argument("--weights",
                        default=CONFIG['eval']['weights_file'],
                        help="starting weights path")
    parser.add_argument("--model",
                        default=CONFIG['eval']['model_file'],
                        help="starting weights path")
    parser.add_argument("--testimg_dir",
                        default=CONFIG['dataset']['test']['images_dir'],
                        help="test image dir path")
    parser.add_argument("--testmsk_dir",
                        default=CONFIG['dataset']['test']['masks_dir'],
                        help="test mask dir path")
    parser.add_argument("--results_dir",
                        default=CONFIG['eval']['results_dir'],
                        help="test mask dir path")
    parser.add_argument("--ground_truth",
                        default=CONFIG['eval']['ground_truth'],
                        help="Compare to ground truth or raw results")
    parser.add_argument("--batch_size",
                        default=CONFIG['eval']['batch_size'],
                        type=int,
                        help="Eval batch size")
    parser.add_argument("--n_labels",
                        default=CONFIG['dataset']['n_labels'],
                        type=int,
                        help="Number of label")
    parser.add_argument("--crop",
                        default=CONFIG['eval']['crop'],
                        help="Crop to input shape, otherwise resize")
    parser.add_argument("--flip",
                        default=CONFIG['eval']['flip'],
                        help="Random flip of training images")
    parser.add_argument("--input_shape",
                        default=CONFIG['segnet']['input_shape'],
                        help="Input images shape")
    parser.add_argument("--kernel",
                        default=CONFIG['segnet']['kernel'],
                        type=int,
                        help="Kernel size")
    parser.add_argument("--pool_size",
                        default=CONFIG['segnet']['pool_size'],
                        help="pooling and unpooling size")
    parser.add_argument("--output_mode",
                        default=CONFIG['segnet']['output_mode'],
                        type=str,
                        help="output activation")
    args = parser.parse_args()
    main(args)
