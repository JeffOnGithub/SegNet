from additional_features import get_n_add_features

"""Configuration information for SegNet"""
CONFIG = dict()

################### Dataset

CONFIG['dataset'] = dict()

#Dataset train
CONFIG['dataset']['train'] = dict()
CONFIG['dataset']['train']['images_dir'] = "./dataset/train/images/"
CONFIG['dataset']['train']['masks_dir'] = "./dataset/train/maps/"

#Dataset val
CONFIG['dataset']['val'] = dict()
CONFIG['dataset']['val']['images_dir'] = "./dataset/val/images/"
CONFIG['dataset']['val']['masks_dir'] = "./dataset/val/maps/"

#Dataset test
CONFIG['dataset']['test'] = dict()
#CONFIG['dataset']['test']['images_dir'] = "./dataset/test/images/"
#CONFIG['dataset']['test']['masks_dir'] = "./dataset/test/maps/"
CONFIG['dataset']['test']['images_dir'] = "./dataset/other_domain/"
CONFIG['dataset']['test']['masks_dir'] = "./dataset/other_domain/"

#Dataset other domain
CONFIG['dataset']['other_domain'] = dict()
CONFIG['dataset']['other_domain']['images_dir'] = "./dataset/other_domain/"

#Dataset properties
CONFIG['dataset']['n_labels'] = 2

################### Training and eval

#Training
CONFIG['training'] = dict()
CONFIG['training']['batch_size'] = 8
CONFIG['training']['n_epochs'] = 10
CONFIG['training']['train_steps'] = 100
CONFIG['training']['val_steps'] = 20
CONFIG['training']['val_steps'] = 1
CONFIG['training']['crop'] = True
CONFIG['training']['flip'] = True
CONFIG['training']['motion_blur'] = 0.25
CONFIG['training']['sp_noise'] = 0.25

#Eval
CONFIG['eval'] = dict()
CONFIG['eval']['batch_size'] = 20
CONFIG['eval']['crop'] = True
CONFIG['eval']['flip'] = True
CONFIG['eval']['ground_truth'] = False
CONFIG['eval']['results_dir'] = "./results/"
CONFIG['eval']['weights_file'] = "./weights/SegNet.hdf5"
CONFIG['eval']['model_file'] = None

################### Segnet properties

CONFIG['segnet'] = dict()
CONFIG['segnet']['cuda_device'] = "1"
CONFIG['segnet']['add_features'] = False
CONFIG['segnet']['input_shape'] = (256, 256, 3)
CONFIG['segnet']['kernel'] = 3
CONFIG['segnet']['pool_size'] = (2, 2)
CONFIG['segnet']['output_mode'] = "softmax"
CONFIG['segnet']['loss'] = "categorical_crossentropy"
CONFIG['segnet']['optimizer'] = "adadelta"
CONFIG['segnet']['reverse_ratio'] = 1

# Add features dimensions if additional features is selected
if CONFIG['segnet']['add_features']:
    CONFIG['segnet']['input_shape'][2] = CONFIG['segnet']['input_shape'][2] + get_n_add_features()