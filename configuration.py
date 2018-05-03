#Configuration information for SegNet
config = dict()

################### Dataset
config['dataset'] = dict()

#Dataset train
config['dataset']['train'] = dict()
config['dataset']['train']['ids_file'] = "./dataset/train/id.txt"
config['dataset']['train']['images_dir'] = "./dataset/train/images/"
config['dataset']['train']['masks_dir'] = "./dataset/train/maps/"

#Dataset val
config['dataset']['val'] = dict()
config['dataset']['val']['ids_file'] = "./dataset/val/id.txt"
config['dataset']['val']['images_dir'] = "./dataset/val/images/"
config['dataset']['val']['masks_dir'] = "./dataset/val/maps/"

#Dataset test
config['dataset']['test'] = dict()
config['dataset']['test']['ids_file'] = "./dataset/test/id.txt"
config['dataset']['test']['images_dir'] = "./dataset/test/images/"
config['dataset']['test']['masks_dir'] = "./dataset/test/maps/"

#Dataset properties
config['dataset']['n_labels'] = 2

################### Training and eval

#Training
config['training'] = dict()
config['training']['batch_size'] = 5
config['training']['n_epochs'] = 1
config['training']['train_steps'] = 100
config['training']['val_steps'] = 20
config['training']['crop'] = True
config['training']['flip'] = True

#Eval
config['eval'] = dict()
config['eval']['batch_size'] = 20
config['eval']['crop'] = True
config['eval']['flip'] = True

################### Segnet properties

config['segnet'] = dict()
config['segnet']['input_shape'] = (256, 256, 3)
config['segnet']['kernel'] = 3
config['segnet']['pool_size'] = (2, 2)
config['segnet']['output_mode'] = "softmax"
config['segnet']['loss'] = "categorical_crossentropy"
config['segnet']['optimizer'] = "adadelta"