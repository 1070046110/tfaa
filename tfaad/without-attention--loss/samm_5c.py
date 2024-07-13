import numpy as np
import copy

import tensorflow as tf
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from attention_block import shuffle_img
from attention_block import path_replace
from attention_block import CreateDs_img
from attention_block import conv_block
from attention_block import f1_score
from attention_block import uar_score
from attention_block import EarlyStopACC
from attention_block import simAM
from attention_block import acc_f1_uar
from attention_block import cbam_attention
from attention_block import se_block


alpha = np.array([1, 1, 1, 1, 1], dtype=np.float32)
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

def attention(x):
    return x


initial_epochs = 200
# 大的batchsize训练集收敛的快，调整的次数少 得到更好的准确率的机会也少
# 小batchsize 训练慢
batch_size = 16
base_learning_rate = 0.001

if __name__ == '__main__':

# dropout 0.5   0.3   0.5
# patience 4     8     4

    for j in range(10):

        accs = []
        f1s = []
        uars = []
        u_sub_imgs = []
        u_sub_labels = []
        v_sub_imgs = []
        v_sub_labels = []

# samm-5c

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

            alpha = tf.reduce_sum(v_train_labels, axis=0)
            num = tf.reduce_sum(v_train_labels) / 5
            alpha = tf.cast(num / alpha, tf.float32)  # alpha是否要归一化 可以试试 权重调整时幅度会更加细腻
            print(alpha)

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
            model.summary()




            model.compile(  # optimizer='adam',
                optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                # loss=rw_ce,
                # loss=focal_loss,
                loss=focal_ce_loss,
                metrics=['accuracy',f1_score,uar_score,acc_f1_uar])
            filepath = 'best_weight_samm_'+str(j+1)+'/weights.best_' + str(i + 1) + 'sub.hdf5'
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc_f1_uar', verbose=1, save_best_only=True,
                                         mode='max')
            reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=10, verbose=1)  # 0.9 3 0.6088

            acc = 0
            f1 = 0
            uar = 0
            early_stop = EarlyStopACC(acc, f1, uar, str(i + 1))

            callbacks_list = [reduce_lr, early_stop, checkpoint]
            # 数据增强
            #   函数方式平衡类别不均   model=sklearn.linear_model.LinearRegression(class_weight='balanced')
            history = model.fit(  # D_train_generate_data(u_train_images,v_train_images,u_train_labels,batch_size),
                [u_train_images, v_train_images],
                u_train_labels,
                # callbacks=[callbacks_list],
                callbacks=callbacks_list,
                steps_per_epoch=len(u_train_images) // batch_size,
                epochs=initial_epochs,
                validation_data=([u_test_images, v_test_images], u_test_labels), batch_size=batch_size)
            accs.append(early_stop.acc)
            f1s.append(early_stop.f1)
            uars.append(early_stop.uar)
            print("acc:"+str(early_stop.acc)+"   "+"f1:"+str(early_stop.f1)+"   "+"uar:"+str(early_stop.uar))

        print("acc:"+str(np.mean(accs))+"   "+"f1:"+str(np.mean(f1s))+"   "+"uar:"+str(np.mean(uars)))
        file = open("best.txt", 'a')
        file.write("samm-5c   acc:"+str(np.mean(accs)) + "   "+"f1:"+str(np.mean(f1s))+ "   "+"uar:"+str(np.mean(uars))+'\n')
        file.close()



