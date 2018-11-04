# -*- coding: utf-8 -*-
"""Evaluate the Segnet model"""

import argparse
from os import listdir, environ
import pandas as pd
import csv
from segnet import create_segnet
from generator import segnet_generator
from configuration import CONFIG

#Set CUDA device for tensorflow
environ["CUDA_VISIBLE_DEVICES"] = CONFIG['segnet']['cuda_device']

def main(args):
    # Eval generator
    eval_list = pd.DataFrame(listdir(args.testimg_dir))
    segnet_train_eval = segnet_generator(img_dir=args.testimg_dir,
                                        mask_dir=args.testmsk_dir,
                                        lists=eval_list,
                                        batch_size=args.batch_size,
                                        dims=[args.input_shape[0], args.input_shape[1]],
                                        n_labels=args.n_labels,
                                        crop=args.crop,
                                        flip=False,
                                        motion_blur=False,
                                        sp_noise=False)

    # Build a network and load weights
    segnet, domain = create_segnet(args.input_shape,
                           args.n_labels,
                           args.kernel,
                           args.pool_size,
                           args.output_mode)
    print("Segnet built")
    
    #Clear result file
    with open(CONFIG['eval']['results_dir'] + "results.csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(['run', 'loss', 'accuracy'])
    
    #Loop all available weights
    for i in range(CONFIG['eval']['weights_array_min'], CONFIG['eval']['weights_array_max'] + 1):
        print(str(i) + " of " + str(CONFIG['eval']['weights_array_max']))
        weights_path = CONFIG['eval']['weights_path']
        segnet.load_weights(weights_path.format(format(i, '02')))
        print("Weights loaded")
        
        # Compile model
        segnet.compile(loss="categorical_crossentropy",
                       optimizer="adadelta",
                       metrics=["accuracy"])
        
        # Run images in the network and get predictions
        result = segnet.evaluate_generator(segnet_train_eval, 
                                            steps=10, 
                                            max_queue_size=10, 
                                            workers=4, 
                                            use_multiprocessing=False, 
                                            verbose=1)
        
        #Save result
        with open(CONFIG['eval']['results_dir'] + "results.csv", 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow([i, result[0], result[1]])


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
