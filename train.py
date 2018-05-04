from SegNet import CreateSegNet
from generator import data_generator
from configuration import config
import argparse
import pandas as pd
from os import listdir

def main(args):
    # set the necessary list
    train_list = pd.DataFrame(listdir(args.trainimg_dir))
    val_list = pd.DataFrame(listdir(args.valimg_dir))

    # set the necessary directories
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir

    train_gen = data_generator(trainimg_dir, trainmsk_dir, train_list, args.batch_size, [args.input_shape[0], args.input_shape[1]], args.n_labels, args.crop, args.flip)
    val_gen = data_generator(valimg_dir, valmsk_dir, val_list, args.batch_size, [args.input_shape[0], args.input_shape[1]], args.n_labels, args.crop, args.flip) 

    segnet = CreateSegNet(args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode)
    print("SegNet created")
    #print(segnet.summary())
    
    # Load weights if specified in args
    if args.weights:
        segnet.load_weights(args.weights)

    segnet.compile(loss=args.loss, optimizer=args.optimizer, metrics=["accuracy"])
    segnet.fit_generator(train_gen, steps_per_epoch=args.epoch_steps, epochs=args.n_epochs, validation_data=val_gen, validation_steps=args.val_steps)

    segnet.save_weights("./weights/SegNet-"+str(args.n_epochs)+".hdf5")
    print("Weights saved")

    #json_string = segnet.to_json()
    #open("./model/SegNet.json", "w").write(json_string)
    #print("Model saved")


if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet dataset")
    parser.add_argument("--weights",
            default=None,
            help="starting weights path")
    parser.add_argument("--trainimg_dir",
            default=config['dataset']['train']['images_dir'],
            help="train image dir path")
    parser.add_argument("--trainmsk_dir",
            default=config['dataset']['train']['masks_dir'],
            help="train mask dir path")
    parser.add_argument("--valimg_dir",
            default=config['dataset']['val']['images_dir'],
            help="val image dir path")
    parser.add_argument("--valmsk_dir",
            default=config['dataset']['val']['masks_dir'],
            help="val mask dir path")
    parser.add_argument("--batch_size",
            default=config['training']['batch_size'],
            type=int,
            help="batch size")
    parser.add_argument("--n_epochs",
            default=config['training']['n_epochs'],
            type=int,
            help="number of epoch")
    parser.add_argument("--epoch_steps",
            default=config['training']['train_steps'],
            type=int,
            help="number of epoch step")
    parser.add_argument("--val_steps",
            default=config['training']['val_steps'],
            type=int,
            help="number of validation step")
    parser.add_argument("--n_labels",
            default=config['dataset']['n_labels'],
            type=int,
            help="Number of label")
    parser.add_argument("--crop",
            default=config['training']['crop'],
            help="Crop to input shape, otherwise resize")
    parser.add_argument("--flip",
            default=config['training']['flip'],
            help="Random flip of training images")
    parser.add_argument("--input_shape",
            default=config['segnet']['input_shape'],
            help="Input images shape")
    parser.add_argument("--kernel",
            default=config['segnet']['kernel'],
            type=int,
            help="Kernel size")
    parser.add_argument("--pool_size",
            default=config['segnet']['pool_size'],
            help="pooling and unpooling size")
    parser.add_argument("--output_mode",
            default=config['segnet']['output_mode'],
            type=str,
            help="output activation")
    parser.add_argument("--loss",
            default=config['segnet']['loss'],
            type=str,
            help="loss function")
    parser.add_argument("--optimizer",
            default=config['segnet']['optimizer'],
            type=str,
            help="optimizer")
    args = parser.parse_args()

    main(args)
