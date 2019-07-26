'''
T2T 버전으로 했을때: log S가 -방향으로 계속 증가, log det W는 항상 +로 오히려 증가, wav range가 계속 증가
Nvidia 버전: log S가 0을 경계로 왔다갔다함. log det W가 -방향으로 천천히 증가, wave range 안정중
'''

import tensorflow as tf
import numpy as np
    
def Nvidia_Invertible_1x1_Conv2D(inputs, reverse=False, name= None, reuse= None):
    with tf.variable_scope(name, 'invertible_1x1_conv2d', reuse= reuse):
        input_Channels = inputs.get_shape().as_list()[-1]

        init_Kernel = np.random.normal(size=(input_Channels, input_Channels)).astype(np.float32);
        if np.linalg.det(init_Kernel) < 0:
            init_Kernel[:, 0] *= -1

        kernel = tf.get_variable('kernel', initializer= init_Kernel)
        kernel = tf.reshape(kernel, [1, 1] + kernel.get_shape().as_list())

        if not reverse:
            '''
            I am not sure why 'tf.linalg.logdet' does not work. Tensorflow displays the following message:
            'Cholesky decomposition was not successful. The input might not be valid. The input might not be valid.'
            '''
            absdet = tf.linalg.det(tf.cast(kernel * 1e+3, tf.float64))
            logdet = tf.cast(tf.log(absdet + 1e-6), tf.float32) - tf.log(1e+3) * input_Channels
            logdet *= tf.cast(tf.shape(inputs)[0] * tf.shape(inputs)[1] * tf.shape(inputs)[2], tf.float32)
            new_Tensor = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], 'SAME', data_format="NHWC")
            return new_Tensor, logdet
        else:            
            new_Tensor = tf.nn.conv2d(inputs, tf.linalg.inv(kernel), [1, 1, 1, 1], 'SAME', data_format="NHWC")
            return new_Tensor
        
def Nvidia_Invertible_1x1_Conv1D(inputs, reverse=False, name= None, reuse= None):
    inputs = tf.expand_dims(inputs, axis=1);
    if not reverse:
        new_Tensor, logdet = Nvidia_Invertible_1x1_Conv2D(inputs, reverse= reverse, name= name, reuse= reuse)
        return tf.squeeze(new_Tensor, axis=1), logdet
    else:
        new_Tensor = Nvidia_Invertible_1x1_Conv2D(inputs, reverse= reverse, name= name, reuse= reuse)
        return tf.squeeze(new_Tensor, axis=1)