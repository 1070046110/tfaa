import tensorflow as tf
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from tensorflow.keras import layers
from attention_block import conv_block
from attention_block import simAM
from attention_block import cbam_attention
from attention_block import se_block


def attention(x,af):
    if af==1:
        x=simAM(x)
    elif af==2:
        x=cbam_attention(x)
    elif af==3:
        x=se_block(x)
    else:
        x=x
    return x


def conv_model(inputs,inputs1,at):

    # inputs
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(24, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 512 28 28
    x_3_1 = conv_block(x, 3, [32, 32, 64], stage=3, block='a-3-1')
    # x_3_1 = identity_block(x_3_1, 3, [32, 32, 64], stage=3, block='b-3-1')

    x_5_1 = conv_block(x, 5, [32, 32, 64], stage=3, block='a-5-1')
    # x_5_1 = identity_block(x_5_1, 5, [32, 32, 64], stage=3, block='b-5-1')

    x_7_1 = conv_block(x, 7, [32, 32, 64], stage=3, block='a-7-1')
    # x_7_1 = identity_block(x_7_1, 7, [32, 32, 64], stage=3, block='b-7-1')

    # inputs1
    x = ZeroPadding2D((3, 3))(inputs1)
    x = Conv2D(24, (7, 7), strides=(2, 2), name='conv2')(x)
    x = BatchNormalization(axis=3, name='bn_conv2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 512 28 28
    x_3_2 = conv_block(x, 3, [32, 32, 64], stage=3, block='a-3-2')
    # x_3_2 = identity_block(x_3_2, 3, [32, 32, 64], stage=3, block='b-3-2')

    x_5_2 = conv_block(x, 5, [32, 32, 64], stage=3, block='a-5-2')
    # x_5_2 = identity_block(x_5_2, 5, [32, 32, 64], stage=3, block='b-5-2')

    x_7_2 = conv_block(x, 7, [32, 32, 64], stage=3, block='a-7-2')
    # x_7_2 = identity_block(x_7_2, 7, [32, 32, 64], stage=3, block='b-7-2')

    x_357_1 = tf.keras.layers.concatenate([x_3_1, x_5_1, x_7_1], axis=-1)
    x_357_1 = attention(x_357_1,at)
    x_357_2 = tf.keras.layers.concatenate([x_3_2, x_5_2, x_7_2], axis=-1)
    x_357_2 = attention(x_357_2,at)

    # 1024 14 14
    x_357_1 = conv_block(x_357_1, 3, [64, 64, 128], stage=4, block='a-3')
    # x_357_1 = identity_block(x_357_1, 3, [64, 64, 128], stage=4, block='b-3')

    x_357_2 = conv_block(x_357_2, 3, [64, 64, 128], stage=4, block='a-5')
    # x_357_2 = identity_block(x_357_2, 5, [64, 64, 128], stage=4, block='b-5')

    # X+ concatenate

    x_357 = tf.keras.layers.concatenate([x_357_1, x_357_2], axis=-1)
    x_357 = attention(x_357,at)

    # 2048 7 7
    x = conv_block(x_357, 3, [128, 128, 256], stage=5, block='a')
    # x = identity_block(x, 3, [128, 128, 256], stage=5, block='b')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    return x

def conv_model_57(inputs,inputs1,at):

    # inputs
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(24, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x1 = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 512 28 28
    # x_3_1 = conv_block(x, 3, [32, 32, 64], stage=3, block='a-3-1')
    # x_3_1 = identity_block(x_3_1, 3, [32, 32, 64], stage=3, block='b-3-1')

    # x_5_1 = conv_block(x, 5, [32, 32, 64], stage=3, block='a-5-1')
    # # x_5_1 = identity_block(x_5_1, 5, [32, 32, 64], stage=3, block='b-5-1')

    # x_7_1 = conv_block(x, 7, [32, 32, 64], stage=3, block='a-7-1')
    # # x_7_1 = identity_block(x_7_1, 7, [32, 32, 64], stage=3, block='b-7-1')

    # inputs1
    x = ZeroPadding2D((3, 3))(inputs1)
    x = Conv2D(24, (7, 7), strides=(2, 2), name='conv2')(x)
    x = BatchNormalization(axis=3, name='bn_conv2')(x)
    x = Activation('relu')(x)
    x2 = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 512 28 28
    # x_3_2 = conv_block(x, 3, [32, 32, 64], stage=3, block='a-3-2')
    # x_3_2 = identity_block(x_3_2, 3, [32, 32, 64], stage=3, block='b-3-2')

    # x_5_2 = conv_block(x, 5, [32, 32, 64], stage=3, block='a-5-2')
    # # x_5_2 = identity_block(x_5_2, 5, [32, 32, 64], stage=3, block='b-5-2')
    #
    # x_7_2 = conv_block(x, 7, [32, 32, 64], stage=3, block='a-7-2')
    # # x_7_2 = identity_block(x_7_2, 7, [32, 32, 64], stage=3, block='b-7-2')


    x_357_1 = attention(x1,at)

    x_357_2 = attention(x2,at)

    # 1024 14 14
    x_357_1 = conv_block(x_357_1, 3, [64, 64, 128], stage=4, block='a-3')
    # x_357_1 = identity_block(x_357_1, 3, [64, 64, 128], stage=4, block='b-3')

    x_357_2 = conv_block(x_357_2, 3, [64, 64, 128], stage=4, block='a-5')
    # x_357_2 = identity_block(x_357_2, 5, [64, 64, 128], stage=4, block='b-5')

    # X+ concatenate

    x_357 = tf.keras.layers.concatenate([x_357_1, x_357_2], axis=-1)
    x_357 = attention(x_357,at)

    # 2048 7 7
    x = conv_block(x_357, 3, [128, 128, 256], stage=5, block='a')
    # x = identity_block(x, 3, [128, 128, 256], stage=5, block='b')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    return x

def conv_model_dp(inputs,inputs1,at,dp):

    # inputs
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(24, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 512 28 28
    x_3_1 = conv_block(x, 3, [32, 32, 64], stage=3, block='a-3-1')
    # x_3_1 = identity_block(x_3_1, 3, [32, 32, 64], stage=3, block='b-3-1')

    x_5_1 = conv_block(x, 5, [32, 32, 64], stage=3, block='a-5-1')
    # x_5_1 = identity_block(x_5_1, 5, [32, 32, 64], stage=3, block='b-5-1')

    x_7_1 = conv_block(x, 7, [32, 32, 64], stage=3, block='a-7-1')
    # x_7_1 = identity_block(x_7_1, 7, [32, 32, 64], stage=3, block='b-7-1')

    # inputs1
    x = ZeroPadding2D((3, 3))(inputs1)
    x = Conv2D(24, (7, 7), strides=(2, 2), name='conv2')(x)
    x = BatchNormalization(axis=3, name='bn_conv2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 512 28 28
    x_3_2 = conv_block(x, 3, [32, 32, 64], stage=3, block='a-3-2')
    # x_3_2 = identity_block(x_3_2, 3, [32, 32, 64], stage=3, block='b-3-2')

    x_5_2 = conv_block(x, 5, [32, 32, 64], stage=3, block='a-5-2')
    # x_5_2 = identity_block(x_5_2, 5, [32, 32, 64], stage=3, block='b-5-2')

    x_7_2 = conv_block(x, 7, [32, 32, 64], stage=3, block='a-7-2')
    # x_7_2 = identity_block(x_7_2, 7, [32, 32, 64], stage=3, block='b-7-2')

    x_357_1 = tf.keras.layers.concatenate([x_3_1, x_5_1, x_7_1], axis=-1)
    x_357_1 = attention(x_357_1,at)
    x_357_2 = tf.keras.layers.concatenate([x_3_2, x_5_2, x_7_2], axis=-1)
    x_357_2 = attention(x_357_2,at)

    # 1024 14 14
    x_357_1 = conv_block(x_357_1, 3, [64, 64, 128], stage=4, block='a-3')
    # x_357_1 = identity_block(x_357_1, 3, [64, 64, 128], stage=4, block='b-3')

    x_357_2 = conv_block(x_357_2, 3, [64, 64, 128], stage=4, block='a-5')
    # x_357_2 = identity_block(x_357_2, 5, [64, 64, 128], stage=4, block='b-5')

    # X+ concatenate

    x_357 = tf.keras.layers.concatenate([x_357_1, x_357_2], axis=-1)
    x_357 = attention(x_357,at)

    # 2048 7 7
    # x = conv_block(x_357, 3, [128, 128, 256], stage=5, block='a')
    x = conv_block(x_357, 3, [128, 128, 256], stage=5, block='a')
    # x = identity_block(x, 3, [128, 128, 256], stage=5, block='b')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dp)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    return x