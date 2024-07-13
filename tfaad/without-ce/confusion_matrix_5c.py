import numpy as np
import copy

import tensorflow as tf
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from tensorflow.keras import layers
from attention_block import shuffle_img
from attention_block import path_replace
from attention_block import CreateDs_img
from attention_block import conv_block
from attention_block import simAM
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def attention(x):
    x=simAM(x)
    return x



if __name__ == '__main__':

# dropout 0.5   0.3   0.5
# patience 4     8     4


    u_sub_imgs = []
    u_sub_labels = []
    v_sub_imgs = []
    v_sub_labels = []
    all_num = []
    kl = tf.zeros([5, 5], dtype=tf.int32)
    zero_one = tf.constant([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
                           dtype=tf.int32)
    f1 = []
    acc = []
    uar = []
    pre = []

    ss=[]
    true_label=[]

# casmeⅡ-5c

    for i in range(27):
        u_all_path = shuffle_img("../data/LOSO_TVL1_SAMM/u/sub" + str(i + 1).zfill(2))
        v_all_path = path_replace(u_all_path, "\\u\\", "\\v\\")
        images, labels = CreateDs_img("../data/LOSO_TVL1_SAMM/u/sub" + str(i + 1).zfill(2), u_all_path)

        u_sub_imgs.append(images)
        u_sub_labels.append(labels)
        images, labels = CreateDs_img("../data/LOSO_TVL1_SAMM/v/sub" + str(i + 1).zfill(2), v_all_path)
        v_sub_imgs.append(images)
        v_sub_labels.append(labels)


    for i in range(27):
        print("第{}折".format(str(i + 1)))
        u_temp_imgs = copy.deepcopy(u_sub_imgs)
        u_temp_labels = copy.deepcopy(u_sub_labels)
        u_test_images = np.vstack(u_temp_imgs[i:i + 1])
        u_test_labels = np.vstack(u_temp_labels[i:i + 1])
        u_temp_imgs.pop(i)
        u_temp_labels.pop(i)
        u_train_images = np.vstack(u_temp_imgs[:])
        u_train_labels = np.vstack(u_temp_labels[:])

        v_temp_imgs = copy.deepcopy(v_sub_imgs)
        v_temp_labels = copy.deepcopy(v_sub_labels)
        v_test_images = np.vstack(v_temp_imgs[i:i + 1])
        v_test_labels = np.vstack(v_temp_labels[i:i + 1])
        v_temp_imgs.pop(i)
        v_temp_labels.pop(i)
        v_train_images = np.vstack(v_temp_imgs[:])
        v_train_labels = np.vstack(v_temp_labels[:])


        # res = ResNet50(include_top=False, weights='imagenet')
        inputs = tf.keras.Input(shape=(224, 224, 1))  # shape不符合resnet输入
        inputs1 = tf.keras.Input(shape=(224, 224, 1))
        # x = preprocess_input(inputs)
        # x1 = preprocess_input(inputs1)

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
        x_357_1 = attention(x_357_1)
        x_357_2 = tf.keras.layers.concatenate([x_3_2, x_5_2, x_7_2], axis=-1)
        x_357_2 = attention(x_357_2)

        # 1024 14 14
        x_357_1 = conv_block(x_357_1, 3, [64, 64, 128], stage=4, block='a-3')
        # x_357_1 = identity_block(x_357_1, 3, [64, 64, 128], stage=4, block='b-3')

        x_357_2 = conv_block(x_357_2, 3, [64, 64, 128], stage=4, block='a-5')
        # x_357_2 = identity_block(x_357_2, 5, [64, 64, 128], stage=4, block='b-5')

        # X+ concatenate

        x_357=tf.keras.layers.concatenate([x_357_1, x_357_2], axis=-1)
        x_357=attention(x_357)

        # 2048 7 7
        x = conv_block(x_357, 3, [128, 128, 256], stage=5, block='a')
        # x = identity_block(x, 3, [128, 128, 256], stage=5, block='b')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
        base_learning_rate = 0.001

        # 训练
        model = tf.keras.Model([inputs, inputs1], outputs=outputs)

        model.load_weights('best_weight_samm_10/weights.best_' + str(i + 1) + 'sub.hdf5')
        predictions = model.predict([u_test_images, v_test_images])
        pred_positive, _ = tf.nn.top_k(predictions, 1)
        pp_out = tf.cast(tf.greater_equal(predictions, pred_positive), tf.float32)
        # matrix = tf.math.confusion_matrix(tf.argmax(u_test_labels,axis=1), tf.argmax(pp_out,axis=1))
        print(tf.argmax(u_test_labels, axis=1))
        print(tf.argmax(pp_out, axis=1))

        ss.append(pp_out)
        true_label.append(u_test_labels)

        # f1.append(f1_score(tf.argmax(u_test_labels, axis=1), tf.argmax(pp_out, axis=1), average='macro'))
        # print(f1_score(tf.argmax(u_test_labels, axis=1), tf.argmax(pp_out, axis=1), average='macro'))
        # print(
        #     recall_score(tf.argmax(u_test_labels, axis=1), tf.argmax(pp_out, axis=1), average='macro', zero_division=1))
        # print(precision_score(tf.argmax(u_test_labels, axis=1), tf.argmax(pp_out, axis=1), average='macro',
        #                       zero_division=1))
        # acc.append(accuracy_score(tf.argmax(u_test_labels, axis=1), tf.argmax(pp_out, axis=1)))
        # uar.append(
        #     recall_score(tf.argmax(u_test_labels, axis=1), tf.argmax(pp_out, axis=1), average='macro', zero_division=0))
        # pre.append(precision_score(tf.argmax(u_test_labels, axis=1), tf.argmax(pp_out, axis=1), average='macro',
        #                            zero_division=0))
        matrix = tf.math.confusion_matrix(
            tf.concat([tf.argmax(u_test_labels, axis=1), tf.constant([4], dtype=tf.int64)], 0),
            tf.concat([tf.argmax(pp_out, axis=1), tf.constant([4], dtype=tf.int64)], 0))
        matrix = matrix - zero_one
        print(matrix)
        # kl=tf.cast(kl,dtype=tf.float32)
        kl = kl + matrix
        all_num.append(tf.reduce_sum(matrix))

    ss=tf.concat(ss,0)
    true_label=tf.concat(true_label,0)
    print('labels shape:')
    print(true_label.shape)
    print('ss shape:')
    print(ss.shape)
    print(tf.reduce_sum(all_num))
    print(kl)
    print(tf.reduce_sum(kl))
    print('f1:')
    print(f1_score(tf.argmax(true_label, axis=1), tf.argmax(ss, axis=1), average='micro'))
    print('uf1:')
    print(f1_score(tf.argmax(true_label, axis=1), tf.argmax(ss, axis=1), average='macro'))
    print('acc:')
    print(accuracy_score(tf.argmax(true_label, axis=1), tf.argmax(ss, axis=1)))
    print('uar:')
    print(recall_score(tf.argmax(true_label, axis=1), tf.argmax(ss, axis=1), average='macro', zero_division=1))
    # print('pre:')
    # print(tf.reduce_mean(pre))





