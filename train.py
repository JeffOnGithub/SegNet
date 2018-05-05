# -*- coding: utf-8 -*-
"""Train the Segnet model"""

import argparse
from os import listdir
import pandas as pd
from segnet import create_segnet
from generator import segnet_generator, domain_generator
from configuration import CONFIG

def main(args):
    """Training"""
    # set the necessary list
    train_list = pd.DataFrame(listdir(args.trainimg_dir))
    val_list = pd.DataFrame(listdir(args.valimg_dir))
    domain_list = pd.DataFrame(listdir(args.domainimg_dir))

    # set the necessary directories
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir
    domainimg_dir = args.domainimg_dir

    # Training generator
    segnet_train_gen = segnet_generator(trainimg_dir,
                               trainmsk_dir,
                               train_list,
                               args.batch_size,
                               [args.input_shape[0], args.input_shape[1]],
                               args.n_labels,
                               args.crop,
                               args.flip,
                               args.motion_blur,
                               args.sp_noise)
    
    # Validation generator
    segnet_val_gen = segnet_generator(valimg_dir,
                             valmsk_dir,
                             val_list,
                             args.batch_size,
                             [args.input_shape[0], args.input_shape[1]],
                             args.n_labels,
                             args.crop,
                             args.flip,
                             args.motion_blur,
                             args.sp_noise) 

    # Domain adaptation generator
    domain_train_gen = domain_generator(trainimg_dir,
                                     domainimg_dir,
                                     train_list,
                                     domain_list,
                                     args.batch_size,
                                     [args.input_shape[0], args.input_shape[1]],
                                     args.crop,
                                     args.flip,
                                     args.motion_blur,
                                     args.sp_noise) 
    
    segnet, domain_adapt = create_segnet(args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode)
    print("SegNet created")
    #print(segnet.summary())
    
    # Load weights if specified in args
    if args.segnet_weights and args.domain_weights:
        segnet.load_weights(args.segnet_weights)
        domain_adapt.load_weights(args.domain_weights)

    segnet.compile(loss=args.loss,
                   optimizer=args.optimizer,
                   metrics=["accuracy"])
    
    domain_adapt.compile(loss=args.loss,
                   optimizer=args.optimizer,
                   metrics=["accuracy"])
    
    for i in range(0, args.n_epochs):
        print("")
        print("--- MAIN EPOCH " + str(i) + " / " + str(args.n_epochs) + " ---")
        print("--- SEGNET")
        segnet.fit_generator(segnet_train_gen,
                             steps_per_epoch=args.epoch_steps,
                             epochs=1,
                             validation_data=segnet_val_gen,
                             validation_steps=args.val_steps)
        print("---  ADAPTATION")
        domain_adapt.fit_generator(domain_train_gen,
                                   steps_per_epoch=args.epoch_steps,
                                   epochs=1)

    segnet.save_weights("./weights/SegNet-"+str(args.n_epochs)+".hdf5")
    domain_adapt.save_weights("./weights/Domain_adapt-"+str(args.n_epochs)+".hdf5")
    print("Weights saved")

if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet dataset")
    parser.add_argument("--segnet_weights",
                        default=None,
                        help="Segnet branch starting weights path")
    parser.add_argument("--domain_weights",
                        default=None,
                        help="Domain adaptation branch starting weights path")
    parser.add_argument("--trainimg_dir",
                        default=CONFIG['dataset']['train']['images_dir'],
                        help="train image dir path")
    parser.add_argument("--trainmsk_dir",
                        default=CONFIG['dataset']['train']['masks_dir'],
                        help="train mask dir path")
    parser.add_argument("--valimg_dir",
                        default=CONFIG['dataset']['val']['images_dir'],
                        help="val image dir path")
    parser.add_argument("--valmsk_dir",
                        default=CONFIG['dataset']['val']['masks_dir'],
                        help="val mask dir path")
    parser.add_argument("--domainimg_dir",
                        default=CONFIG['dataset']['other_domain']['images_dir'],
                        help="domain image dir path")
    parser.add_argument("--batch_size",
                        default=CONFIG['training']['batch_size'],
                        type=int,
                        help="batch size")
    parser.add_argument("--n_epochs",
                        default=CONFIG['training']['n_epochs'],
                        type=int,
                        help="number of epoch")
    parser.add_argument("--epoch_steps",
                        default=CONFIG['training']['train_steps'],
                        type=int,
                        help="number of epoch step")
    parser.add_argument("--val_steps",
                        default=CONFIG['training']['val_steps'],
                        type=int,
                        help="number of validation step")
    parser.add_argument("--n_labels",
                        default=CONFIG['dataset']['n_labels'],
                        type=int,
                        help="Number of label")
    parser.add_argument("--crop",
                        default=CONFIG['training']['crop'],
                        help="Crop to input shape, otherwise resize")
    parser.add_argument("--flip",
                        default=CONFIG['training']['flip'],
                        help="Random flip of training images")
    parser.add_argument("--sp_noise",
                        default=CONFIG['training']['sp_noise'],
                        help="Fraction of images with added noise")
    parser.add_argument("--motion_blur",
                        default=CONFIG['training']['motion_blur'],
                        help="Fraction of images with added blur")
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
    parser.add_argument("--loss",
                        default=CONFIG['segnet']['loss'],
                        type=str,
                        help="loss function")
    parser.add_argument("--optimizer",
                        default=CONFIG['segnet']['optimizer'],
                        type=str,
                        help="optimizer")
    args = parser.parse_args()
    main(args)
