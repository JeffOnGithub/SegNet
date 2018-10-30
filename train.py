# -*- coding: utf-8 -*-
"""Train the Segnet model"""

import argparse
from os import listdir, environ
import pandas as pd
from keras.callbacks import ModelCheckpoint
from segnet import create_segnet
from generator import segnet_generator, domain_generator
from configuration import CONFIG

#Set CUDA device for tensorflow
environ["CUDA_VISIBLE_DEVICES"] = CONFIG['segnet']['cuda_device']

def main(args):
    """Training"""
    
    training_list = pd.DataFrame(listdir(args.trainimg_dir))
    val_list = pd.DataFrame(listdir(args.valimg_dir))
    #from sklearn.model_selection import train_test_split
    #training_list, val_list = train_test_split(pd.DataFrame(listdir(args.trainimg_dir)), test_size=0.1)
    
    # Training generator
    segnet_train_gen = segnet_generator(img_dir=args.trainimg_dir,
                                        mask_dir=args.trainmsk_dir,
                                        lists=training_list,
                                        batch_size=args.batch_size,
                                        dims=[args.input_shape[0], args.input_shape[1]],
                                        n_labels=args.n_labels,
                                        crop=args.crop,
                                        flip=args.flip,
                                        motion_blur=args.motion_blur,
                                        sp_noise=args.sp_noise)
    
    # Validation generator
    segnet_val_gen = segnet_generator(img_dir=args.valimg_dir,
                                        mask_dir=args.valmsk_dir,
                                        lists=val_list,
                                        batch_size=args.batch_size,
                                        dims=[args.input_shape[0], args.input_shape[1]],
                                        n_labels=args.n_labels,
                                        crop=args.crop,
                                        flip=args.flip,
                                        motion_blur=args.motion_blur,
                                        sp_noise=args.sp_noise)

    # Domain adaptation generator
    domain_train_gen = domain_generator(img_dir=args.trainimg_dir,
                                        domain_dir=args.domainimg_dir,
                                        img_list=pd.DataFrame(listdir(args.trainimg_dir)),
                                        domain_list=pd.DataFrame(listdir(args.domainimg_dir)),
                                        batch_size=args.batch_size,
                                        dims=[args.input_shape[0], args.input_shape[1]],
                                        crop=args.crop,
                                        flip=args.flip,
                                        motion_blur=args.motion_blur,
                                        sp_noise=args.sp_noise) 
    
    # Create the complete network
    segnet, domain_adapt = create_segnet(input_shape=args.input_shape,
                                         n_labels=args.n_labels,
                                         kernel=args.kernel,
                                         pool_size=args.pool_size,
                                         output_mode=args.output_mode,
                                         reverse_ratio=args.reverse_ratio)
    print("SegNet/DANN created")
    
    # Load weights if specified in args
    if args.segnet_weights and args.domain_weights:
        segnet.load_weights(args.segnet_weights)
        domain_adapt.load_weights(args.domain_weights)

    # Compile both models
    segnet.compile(loss=args.loss,
                   optimizer=args.optimizer,
                   metrics=["accuracy"])
    domain_adapt.compile(loss=args.loss,
                         optimizer=args.optimizer,
                         metrics=["accuracy"])
    
    #Set callbacks
    checkpoint = ModelCheckpoint(filepath="./weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
                                 monitor='val_acc', 
                                 verbose=1, 
                                 save_best_only=False, 
                                 save_weights_only=True)
    callbacks_list = [checkpoint]
    
    # Custom training loop
    # Each complete epoch is one epoch on segnet, one epoch on dann
    for i in range(0, args.n_epochs):
        print("")
        print("--- MAIN EPOCH " + str(i + 1) + " / " + str(args.n_epochs) + " ---")
        print("--- SEGNET ---")
        segnet.fit_generator(segnet_train_gen,
                             steps_per_epoch=args.epoch_steps,
                             epochs=i+1,
                             initial_epoch=i,
                             validation_data=segnet_val_gen,
                             validation_steps=args.val_steps,
                             workers=2,
                             max_queue_size=2 * args.batch_size,
                             callbacks=callbacks_list)
                             #class_weight = args.class_weight)
        print("---  ADAPTATION ---")
        domain_adapt.fit_generator(domain_train_gen,
                                   steps_per_epoch=args.epoch_steps / 2,
                                   epochs=1,
                                   workers=2,
                                   max_queue_size=2 * args.batch_size)
            
    # Save weights of both models on completion
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
    parser.add_argument("--class_weight",
                        default=CONFIG['dataset']['class_weight'],
                        help="Weight of segmentation classes")
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
    parser.add_argument("--reverse_ratio",
                        default=CONFIG['segnet']['reverse_ratio'],
                        type=int,
                        help="Gradient multiplier for the dann branch")
    args = parser.parse_args()
    main(args)
