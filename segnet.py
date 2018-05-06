# -*- coding: utf-8 -*-
"""Build the Segnet model"""

from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model
from keras.layers.core import Activation, Reshape, Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

from keras_MaxPoolingLayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras_GradientReversalLayer import GradientReversal

def full_conv2D_layer(input_tensor,
                      n_features,
                      kernel,
                      padding,
                      activation):
    """Full stack of conv2D layers"""
    output = Convolution2D(n_features,
                           (kernel, kernel),
                           padding=padding)(input_tensor)
    output = BatchNormalization()(output)
    output = Activation(activation)(output)
    return output

def full_conv2D_block(x, n_layers, n_features, kernel=3, padding="same", activation="relu"):
    """Create a block of full stack conv2D layers"""
    for i in range(0, n_layers):
        x = full_conv2D_layer(x, n_features, kernel, padding, activation)
    return x

def create_segnet(input_shape,
                  n_labels,
                  kernel,
                  pool_size=(2, 2),
                  output_mode="softmax",
                  reverse_ratio=1):
    """Create a segnet model and returns it"""
    
    # Encoder
    inputs = Input(shape=input_shape)

    pool_block_1 = full_conv2D_block(x=inputs, n_layers=2, n_features=64, kernel=kernel)
    
    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(pool_block_1)
    
    pool_block_2 = full_conv2D_block(x=pool_1, n_layers=2, n_features=128, kernel=kernel)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(pool_block_2)

    pool_block_3 = full_conv2D_block(x=pool_2, n_layers=2, n_features=256, kernel=kernel)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(pool_block_3)
    
    pool_block_4 = full_conv2D_block(x=pool_3, n_layers=2, n_features=512, kernel=kernel)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(pool_block_4)

    pool_block_5 = full_conv2D_block(x=pool_4, n_layers=2, n_features=512, kernel=kernel)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(pool_block_5)

    # Decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    unpool_block_1 = full_conv2D_block(x=unpool_1, n_layers=2, n_features=512, kernel=kernel)

    unpool_2 = MaxUnpooling2D(pool_size)([unpool_block_1, mask_4])

    unpool_block_2 = full_conv2D_block(x=unpool_2, n_layers=2, n_features=256, kernel=kernel)

    unpool_3 = MaxUnpooling2D(pool_size)([unpool_block_2, mask_3])

    unpool_block_3 = full_conv2D_block(x=unpool_3, n_layers=2, n_features=128, kernel=kernel)

    unpool_4 = MaxUnpooling2D(pool_size)([unpool_block_3, mask_2])

    unpool_block_4 = full_conv2D_block(x=unpool_4, n_layers=2, n_features=64, kernel=kernel)

    unpool_5 = MaxUnpooling2D(pool_size)([unpool_block_4, mask_1])

    unpool_block_5 = full_conv2D_block(x=unpool_5, n_layers=1, n_features=64, kernel=kernel)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(unpool_block_5)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

    main_output = Activation(output_mode, name='main_output')(conv_26)

    # Domain branch
    # Flatten incoming convolutions for dense layers
    domain_0 = Flatten()(pool_block_5)
    # Flip gradient on backpropagation (DANN)
    domain_0 = GradientReversal(reverse_ratio)(domain_0)
    # Standard dense layers
    domain_1 = Dense(128, activation='relu')(domain_0)
    domain_2 = Dense(64, activation='relu')(domain_1)
    domain_3 = Dense(64, activation='relu')(domain_2)
    aux_output = Dense(2, activation='sigmoid', name='aux_output')(domain_3)
    
    segnet = Model(inputs=inputs, outputs=main_output, name="SegNet")
    domain_adapt = Model(inputs=inputs, outputs=aux_output, name="Domain_adaptation")

    return segnet, domain_adapt

#Unit testing
#print('Segnet creation started')
#model1, model2 = create_segnet((128, 128, 3), 2, kernel=3, pool_size=(2, 2), output_mode="softmax")
#print('Segnet created')
#plot_model(model1, to_file='./model/structure1.png', show_shapes=True, show_layer_names=True)
#plot_model(model2, to_file='./model/structure2.png', show_shapes=True, show_layer_names=True)
