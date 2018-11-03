"""Configuration information for SegNet"""
CONFIG = dict()

################### Dataset

CONFIG['dataset'] = dict()

#Dataset train
CONFIG['dataset']['train'] = dict()
CONFIG['dataset']['train']['images_dir'] = "Z:/transfert_test/train_syn/images/"
CONFIG['dataset']['train']['masks_dir'] = "Z:/transfert_test/train_syn/maps_binary_roads/"

#Dataset val
CONFIG['dataset']['val'] = dict()
CONFIG['dataset']['val']['images_dir'] = "Z:/transfert_test/val_syn/images/"
CONFIG['dataset']['val']['masks_dir'] = "Z:/transfert_test/val_syn/maps_binary_roads/"

#Dataset test
CONFIG['dataset']['test'] = dict()
CONFIG['dataset']['test']['images_dir'] = "Z:/transfert_test/test_real/images_resized/"
CONFIG['dataset']['test']['masks_dir'] = "Z:/transfert_test/test_real/maps_resized_binary_roads/"
#CONFIG['dataset']['test']['images_dir'] = "Z:/dataset/other_domain/"
#CONFIG['dataset']['test']['masks_dir'] = "Z:/dataset/other_domain/"

#Dataset other domain
CONFIG['dataset']['other_domain'] = dict()
CONFIG['dataset']['other_domain']['images_dir'] = "Z:/transfert_test/train_real/images_resized/"

#Dataset properties
CONFIG['dataset']['n_labels'] = 2
CONFIG['dataset']['class_weight'] = {0: 1.0, 1: 1.0}

################### Training and eval

#Training
CONFIG['training'] = dict()
CONFIG['training']['batch_size'] = 4
CONFIG['training']['n_epochs'] = 25
CONFIG['training']['train_steps'] = 100
CONFIG['training']['val_steps'] = 10
CONFIG['training']['crop'] = False
CONFIG['training']['flip'] = True
CONFIG['training']['motion_blur'] = 0.25
CONFIG['training']['sp_noise'] = 0.25
CONFIG['training']['domain_adaptation'] = False

#Eval
CONFIG['eval'] = dict()
CONFIG['eval']['batch_size'] = 30
CONFIG['eval']['crop'] = False
CONFIG['eval']['flip'] = False
CONFIG['eval']['ground_truth'] = True
CONFIG['eval']['results_dir'] = "./results/"
CONFIG['eval']['weights_file'] = "./weights/SegNet.hdf5"
CONFIG['eval']['weights_path'] = "./weights/weights.{}.hdf5"
CONFIG['eval']['weights_array_min'] = 1
CONFIG['eval']['weights_array_max'] = 25
CONFIG['eval']['model_file'] = None

################### Segnet properties

CONFIG['segnet'] = dict()
CONFIG['segnet']['cuda_device'] = "0"
CONFIG['segnet']['input_shape'] = (384, 384, 3)
CONFIG['segnet']['kernel'] = 3
CONFIG['segnet']['pool_size'] = (2, 2)
CONFIG['segnet']['output_mode'] = "softmax"
CONFIG['segnet']['loss'] = "categorical_crossentropy"
CONFIG['segnet']['optimizer'] = "adadelta"
CONFIG['segnet']['reverse_ratio'] = 1
