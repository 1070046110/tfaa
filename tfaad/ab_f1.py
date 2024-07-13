import keras.callbacks
import numpy as np

import tensorflow as tf
from keras import layers
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import BatchNormalization
import keras.backend as K
import pathlib
import random
from PIL import Image
from tensorflow.keras.layers import Lambda
from tensorflow.keras import layers




def to_onehot(b):
    a = dict((name, index) for index, name in enumerate(b))
    for i in range(len(b)):
        c = np.zeros(len(b))
        c[i] = 1
        a[b[i]] = c
    return a


def shuffle_img(path):
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    return all_image_paths


def path_replace(all_paths, str_old, str_new):
    all_paths = [path.replace(str_old, str_new) for path in all_paths]
    return all_paths


def CreateDs_img(path, all_image_paths):
    data_root = pathlib.Path(path)
    image_count = len(all_image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = to_onehot(label_names)
    category = 5
    input_dt = (224, 224, 3)
    test_images_list = []
    test_label_list = []
    for i in range(image_count):
        curr_img = Image.open(all_image_paths[i])
        curr_img = curr_img.resize([input_dt[0], input_dt[1]])
        curr_img_array = np.array(curr_img).astype(np.uint8)

        # curr_img_array = cv2.cvtColor(curr_img_array,
        #                               cv2.COLOR_GRAY2RGB)  ###########################灰阶图装rgb############################################

        test_images_list.append(curr_img_array)
        curr_label = label_to_index[pathlib.Path(all_image_paths[i]).parent.name]
        test_label_list.append(curr_label)
    images_array = np.array(test_images_list)
    labels_array = np.array(test_label_list)
    return images_array, labels_array


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)


    x = Activation('relu')(x)
    return x



# 自定义早停
class EarlyStopACC(keras.callbacks.Callback):
    def __init__(self, acc, f1, uar, str):
        self.acc = acc
        self.f1 = f1
        self.str = str
        self.uar = uar

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        val_f1 = logs.get('val_f11_score')
        val_uar = logs.get('val_uar1_score')
        if self.acc < val_acc:
            self.acc = val_acc
            self.f1 = val_f1
            self.uar = val_uar
        elif self.acc == val_acc:
            if self.f1 < val_f1:
                self.f1 = val_f1
            if self.uar < val_uar:
                self.uar = val_uar
        print("第{}折".format(self.str) + "  acc:" + str(self.acc) + "  f1:" + str(self.f1) + "  uar:" + str(self.uar))
        if val_acc == 1 and val_f1 >= 0.99999 and val_uar >= 0.99999:
            self.model.stop_training = True


alpha = np.array([1, 1, 1, 1, 1], dtype=np.float32)


def rw_ce(y_true, y_pred):
    # loss=-tf.reduce_sum(alpha*y_true*tf.math.log(y_pred+0.0001)+alpha*(1.-y_true)*tf.math.log(1.-y_pred+0.0001),axis=1)

    loss = -tf.reduce_sum(
        y_true * tf.math.log(y_pred + 0.0001) * alpha,
        axis=1)#0是列，1是行
    loss = tf.reduce_mean(loss)
    return loss


def focal_loss(y_true, y_pred):
    loss = -tf.reduce_sum(
        y_true * alpha * tf.pow(1. - y_pred + 0.0001, 2) * tf.math.log(y_pred + 0.0001),
        axis=1)
    loss = tf.reduce_mean(loss)
    return loss

a=0.3
def focal_ce_loss(y_true, y_pred):
    loss_ce = -tf.reduce_sum(
        y_true * tf.math.log(y_pred + 0.0001),
        axis=1)  # 0是列，1是行
    loss_ce = tf.reduce_mean(loss_ce)

    loss_focal = -tf.reduce_sum(
        y_true * alpha * tf.pow(1. - y_pred + 0.0001, 2) * tf.math.log(y_pred + 0.0001),
        axis=1)
    loss_focal = tf.reduce_mean(loss_focal)

    loss=((1-a)*loss_ce)+(a*loss_focal)
    return loss

def return_y_true(y_true, y_pred):
    return y_true

def return_y_pred(y_true, y_pred):
    return y_pred

def f11_score(y_true,y_pred):
    pred_positive,_=tf.nn.top_k(y_pred,1)
    pp_out=tf.cast(tf.greater_equal(y_pred,pred_positive),tf.float32)
    all_pred_positive=tf.reduce_sum(pp_out,axis=0)#自己预测的正类
    pred_positive_true=y_true*pp_out
    ppt=tf.reduce_sum(pred_positive_true,axis=0)#预测对的正类
    true_positive=tf.reduce_sum(y_true,axis=0)#真正的正类
    p=ppt/(all_pred_positive+0.0000001)
    r=ppt/(true_positive+0.0000001)

    f1=2*p*r/(p+r+0.0000001)

    zero_num=tf.reduce_sum(tf.where(tf.equal(true_positive,0),tf.ones_like(f1),tf.zeros_like(f1)))
    val_all=tf.reduce_sum(f1)
    f1=float(val_all)/(f1.shape[0]-float(zero_num))

    return f1


def uar1_score(y_true, y_pred):
    pred_positive, _ = tf.nn.top_k(y_pred, 1)
    pp_out = tf.cast(tf.greater_equal(y_pred, pred_positive), tf.float32)
    pred_positive_true = y_true * pp_out
    ppt = tf.reduce_sum(pred_positive_true, axis=0)  # 预测对的正类
    true_positive = tf.reduce_sum(y_true, axis=0)  # 真正的正类

    r = ppt / (true_positive + 0.0000001)

    zero_num = tf.reduce_sum(tf.where(tf.equal(true_positive, 0), tf.ones_like(r), tf.zeros_like(r)))
    val_all = tf.reduce_sum(r)

    uar=float(val_all)/(r.shape[0]-float(zero_num))


    return uar

def acc_f1_uar(y_true, y_pred):
    pred_positive, _ = tf.nn.top_k(y_pred, 1)
    pp_out = tf.cast(tf.greater_equal(y_pred, pred_positive), tf.float32)
    all_pred_positive = tf.reduce_sum(pp_out, axis=0)  # 自己预测的正类
    pred_positive_true = y_true * pp_out
    ppt = tf.reduce_sum(pred_positive_true, axis=0)  # 预测对的正类
    true_positive = tf.reduce_sum(y_true, axis=0)  # 真正的正类
    p = ppt / (all_pred_positive + 0.0000001)

    r = ppt / (true_positive + 0.0000001)

    zero_num = tf.reduce_sum(tf.where(tf.equal(true_positive, 0), tf.ones_like(r), tf.zeros_like(r)))
    val_all = tf.reduce_sum(r)

    uar=float(val_all)/(r.shape[0]-float(zero_num))

    f1 = 2 * p * r / (p + r + 0.0000001)

    zero_num = tf.reduce_sum(tf.where(tf.equal(true_positive, 0), tf.ones_like(f1), tf.zeros_like(f1)))
    val_all = tf.reduce_sum(f1)
    f1 = float(val_all) / (f1.shape[0] - float(zero_num))

    acc=tf.reduce_sum(ppt)/tf.reduce_sum(y_true)
    acc=float(acc)

    return 100*acc+f1+uar



def se_block(inputs, ratio=4):  # ratio代表第一个全连接层下降通道数的系数

    # 获取输入特征图的通道数
    in_channel = inputs.shape[-1]

    # 全局平均池化[h,w,c]==>[None,c]
    x = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[1,1,c]
    x = layers.Reshape(target_shape=(1, 1, in_channel))(x)

    # [1,1,c]==>[1,1,c/4]
    x = layers.Dense(in_channel // ratio)(x)  # 全连接下降通道数

    # relu激活
    x = tf.nn.relu(x)

    # [1,1,c/4]==>[1,1,c]
    x = layers.Dense(in_channel)(x)  # 全连接上升通道数

    # sigmoid激活，权重归一化
    x = tf.nn.sigmoid(x)

    # [h,w,c]*[1,1,c]==>[h,w,c]
    outputs = layers.multiply([inputs, x])  # 归一化权重和原输入特征图逐通道相乘

    return outputs


# （1）通道注意力
def channel_attenstion(inputs, ratio=0.25):
    '''ratio代表第一个全连接层下降通道数的倍数'''

    channel = inputs.shape[-1]  # 获取输入特征图的通道数

    # 分别对输出特征图进行全局最大池化和全局平均池化
    # [h,w,c]==>[None,c]
    x_max = layers.GlobalMaxPooling2D()(inputs)
    x_avg = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[1,1,c]
    x_max = layers.Reshape([1, 1, -1])(x_max)  # -1代表自动寻找通道维度的大小
    x_avg = layers.Reshape([1, 1, -1])(x_avg)  # 也可以用变量channel代替-1

    # 第一个全连接层通道数下降1/4, [1,1,c]==>[1,1,c//4]
    x_max = layers.Dense(channel * ratio)(x_max)
    x_avg = layers.Dense(channel * ratio)(x_avg)

    # relu激活函数
    x_max = layers.Activation('relu')(x_max)
    x_avg = layers.Activation('relu')(x_avg)

    # 第二个全连接层上升通道数, [1,1,c//4]==>[1,1,c]
    x_max = layers.Dense(channel)(x_max)
    x_avg = layers.Dense(channel)(x_avg)

    # 结果在相叠加 [1,1,c]+[1,1,c]==>[1,1,c]
    x = layers.Add()([x_max, x_avg])

    # 经过sigmoid归一化权重
    x = tf.nn.sigmoid(x)

    # 输入特征图和权重向量相乘，给每个通道赋予权重
    x = layers.Multiply()([inputs, x])  # [h,w,c]*[1,1,c]==>[h,w,c]

    return x


# （2）空间注意力机制
def spatial_attention(inputs):
    # 在通道维度上做最大池化和平均池化[b,h,w,c]==>[b,h,w,1]
    # keepdims=Fale那么[b,h,w,c]==>[b,h,w]
    x_max = tf.reduce_max(inputs, axis=3, keepdims=True)  # 在通道维度求最大值
    x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)  # axis也可以为-1

    # 在通道维度上堆叠[b,h,w,2]
    x = layers.concatenate([x_max, x_avg])

    # 1*1卷积调整通道[b,h,w,1]
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)

    # sigmoid函数权重归一化
    x = tf.nn.sigmoid(x)

    # 输入特征图和权重相乘
    x = layers.Multiply()([inputs, x])

    return x


# （3）CBAM注意力
def cbam_attention(inputs):
    # 先经过通道注意力再经过空间注意力
    x = channel_attenstion(inputs)
    x = spatial_attention(x)
    return x





def simAM(inputs, lambd=1e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_dim = K.int_shape(inputs)
    if channel_axis == -1:
        n = in_dim[1]*in_dim[2]-1
        d = Lambda(lambda x: K.pow(x - K.mean(x, axis=[1, 2], keepdims=True), 2))(inputs)
        v = Lambda(lambda d: K.sum(d, axis=[1, 2], keepdims=True) / n)(d)
        E_inv = Lambda(lambda x: x[0] / (4*(x[1]+lambd)) + 0.5)([d, v])
    else:
        n = in_dim[2] * in_dim[3] - 1
        d = Lambda(lambda x: K.pow(x - K.mean(x, axis=[2, 3], keepdims=True), 2))(inputs)
        v = Lambda(lambda d: K.sum(d, axis=[2, 3], keepdims=True) / n)(d)
        E_inv = Lambda(lambda x: x[0] / (4 * (x[1] + lambd)) + 0.5)([d, v])
    return tf.multiply(inputs, Activation('sigmoid')(E_inv))


def attention(x):
    x=simAM(x)
    return x
