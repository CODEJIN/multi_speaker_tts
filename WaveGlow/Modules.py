#Reference
#https://github.com/b04901014/waveglow-tensorflow/blob/master/src/module.py

import tensorflow as tf;
import numpy as np;
import Hyper_Parameters as hp
from WaveGlow.Inv1x1 import Nvidia_Invertible_1x1_Conv1D as Invertible_1x1_Conv1D

def Get_Weight_Norm_Variable(
    name,
    shape=None,
    dtype=tf.float32,
    initializer=None,
    trainable=True
    ):  
    with tf.name_scope(name) as scope:          
        g = tf.get_variable(
            name='g',
            shape=shape[-1:],
            dtype=dtype,
            initializer=initializer,
            trainable=trainable
            )
        v = tf.get_variable(
            name='kernel',
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            trainable=trainable
            )
        v_Norm = tf.nn.l2_normalize(v, axis=[0, 1, 2], epsilon=1e-5, name='norm_kernel')

        return g * v_Norm

def Weight_Norm_Conv2D(
    inputs,
    filters,
    kernel_size,
    strides= [1, 1, 1, 1],
    padding= 'valid',
    data_format= 'NHWC', #or 'NCHW'
    dilations= [1, 1, 1, 1],
    activation= None,
    use_bias= True,
    kernel_initializer= None,
    bias_initializer= tf.zeros_initializer(),
    trainable= True,
    name= None
    ):
    with tf.variable_scope(name, 'weight_norm_conv2d') as scope:
        if type(kernel_size) == int:
            kernel_size = [kernel_size, kernel_size]

        norm_Kernel = Get_Weight_Norm_Variable(
            name= name,
            shape= kernel_size + [inputs.get_shape()[-1], filters],
            dtype= inputs.dtype,
            initializer= kernel_initializer,
            trainable= trainable
            )
        new_Tensor = tf.nn.conv2d(
            input= inputs,
            filter= norm_Kernel,
            strides= strides,
            padding = padding.upper(),
            data_format=data_format.upper(),
            dilations= dilations,
            name='conv2d'
            )
        if use_bias:
            bias = tf.get_variable(
                name= 'bias',
                shape= [filters],
                dtype= inputs.dtype,
                initializer= bias_initializer,
                trainable= trainable
                )
            new_Tensor = tf.add(new_Tensor, bias, name='bias_add')

        if activation is not None:
            new_Tensor = activation(new_Tensor)

        return new_Tensor

def Weight_Norm_Conv1D(
    inputs,
    filters,
    kernel_size,
    strides= 1,
    padding= 'valid',
    data_format= 'NWC', #or 'NCW'
    dilations= 1,
    activation= None,
    use_bias= True,
    kernel_initializer= None,
    bias_initializer= tf.zeros_initializer(),
    trainable= True,
    name= None
    ):    
    data_format = data_format.upper().replace('W', 'HW');
    kernel_size = [1, kernel_size];

    if data_format == 'NHWC':
        inputs = tf.expand_dims(inputs, axis=1)
        strides = [1, 1, strides, 1];
        dilations= [1, 1, dilations, 1];
    elif data_format == 'NCHW':
        inputs = tf.expand_dims(inputs, axis=2)
        strides = [1, 1, 1, strides];
        dilations= [1, 1, 1, dilations];

    new_Tensor = Weight_Norm_Conv2D(
        inputs= inputs,
        filters= filters,
        kernel_size= kernel_size,
        strides= strides,
        padding= padding,
        data_format= data_format, #or 'NCHW'
        dilations= dilations,
        activation= activation,
        use_bias= use_bias,
        kernel_initializer= kernel_initializer,
        bias_initializer= bias_initializer,
        trainable= trainable,
        name= name
        )

    if data_format == 'NHWC':
        new_Tensor = tf.squeeze(new_Tensor, axis=1)
    elif data_format == 'NCHW':
        new_Tensor = tf.squeeze(new_Tensor, axis=2)

    return new_Tensor

def Restructure_Train_Data(audios, mels):
    new_Audio_Tensor = tf.slice(
        audios,
        begin=[0, 0],
        size=[
            tf.shape(audios)[0],
            tf.cast(tf.shape(audios)[1] / hp.WaveGlow.Groups, tf.int32) * tf.cast(hp.WaveGlow.Groups, tf.int32)
            ]
        )   #[N, T, S]

    new_Mel_Tensor = Upsample_Mel(mels);
    
    new_Mel_Tensor = tf.slice(
        new_Mel_Tensor,
        begin=[0, 0, 0],
        size=[
            tf.shape(new_Mel_Tensor)[0],
            tf.shape(new_Audio_Tensor)[1],
            new_Mel_Tensor.get_shape()[2]
            ]
        )   #[N, T, C]

    new_Mel_Tensor = tf.reshape(
        new_Mel_Tensor,
        shape=[
            -1,
            tf.cast(tf.shape(new_Mel_Tensor)[1] / hp.WaveGlow.Groups, tf.int32),
            hp.WaveGlow.Groups * hp.Sound.Mel_Dim
            ]
        )   #[N, T/G, G*C]
        
    new_Audio_Tensor = tf.reshape(
        new_Audio_Tensor,
        shape=[
            -1,
            tf.cast(tf.shape(new_Audio_Tensor)[1] / hp.WaveGlow.Groups, tf.int32),
            hp.WaveGlow.Groups
            ]
        )   #[N, T/G, G*C]        

    return new_Audio_Tensor, new_Mel_Tensor

def Restructure_Inference_Data(mels):
    new_Mel_Tensor = Upsample_Mel(mels);
    #new_Mel_Tensor = new_Mel_Tensor[:,:-(hp.WaveGlow.Upsample.Kernel_Size - hp.WaveGlow.Upsample.Strides),:]
    new_Mel_Tensor = tf.reshape(
        new_Mel_Tensor,
        shape=[
            -1,
            tf.cast(tf.shape(new_Mel_Tensor)[1] / hp.WaveGlow.Groups, tf.int32),
            hp.WaveGlow.Groups * hp.Sound.Mel_Dim
            ]
        )   #[N, T/G, G*C]

    new_Audio_Tensor = tf.random.normal((
        tf.shape(new_Mel_Tensor)[0],
        tf.shape(new_Mel_Tensor)[1],
        hp.WaveGlow.Groups - (np.ceil(hp.WaveGlow.Flows / hp.WaveGlow.Early_Every) - 1) * hp.WaveGlow.Early_Size
        ))

    return new_Audio_Tensor, new_Mel_Tensor

#https://github.com/chrisdonahue/wavegan/blob/master/wavegan.py
def Upsample_Mel(inputs):
    inputs = tf.expand_dims(inputs, axis=1)
    conv2d_Transpose_Tensor = tf.layers.conv2d_transpose(
        inputs= inputs,
        filters= hp.Sound.Mel_Dim,
        kernel_size= (1, hp.WaveGlow.Upsample.Kernel_Size),
        strides=(1, hp.WaveGlow.Upsample.Strides),
        kernel_initializer= tf.initializers.random_uniform(0, 0.02),
        bias_initializer= tf.initializers.zeros(),
        )    
    return tf.squeeze(conv2d_Transpose_Tensor, axis=1)

def Affine_Coupling_Layer(audio_Tensor, mel_Tensor, reverse= False, name= None, reuse= None):
    with tf.variable_scope(name, 'affine_coupling_layer', reuse= reuse):
        if not reverse:
            audio_Tensor, log_Det_W_Tensor = Invertible_1x1_Conv1D(
                inputs= audio_Tensor,
                reverse= reverse,
                name= 'invertible_1x1',
                reuse= tf.AUTO_REUSE
                )

        audio0_Tensor, audio1_Tensor = tf.split(
            audio_Tensor,
            num_or_size_splits= 2,
            axis=-1
            )
        
        log_S_Tensor, bias_Tensor = WaveNet(
            audio_Tensor= audio0_Tensor,
            mel_Tensor= mel_Tensor,
            name='wavenet',
            reuse= reuse
            )

        if not reverse:
            log_S_Tensor  = tf.minimum(log_S_Tensor, 8.0)   #To avoid inf
            audio1_Tensor = tf.exp(log_S_Tensor) * audio1_Tensor + bias_Tensor;

            new_Audio_Tensor = tf.concat([audio0_Tensor, audio1_Tensor], axis=-1)

            return new_Audio_Tensor, tf.reduce_sum(log_S_Tensor), tf.squeeze(log_Det_W_Tensor)        
        else:
            audio1_Tensor = (audio1_Tensor - bias_Tensor) / tf.exp(log_S_Tensor)

            new_Audio_Tensor = tf.concat([audio0_Tensor, audio1_Tensor], axis=-1)
            new_Audio_Tensor = Invertible_1x1_Conv1D(
                inputs= new_Audio_Tensor,
                reverse= reverse,
                name= 'invertible_1x1',
                reuse= tf.AUTO_REUSE
                )
            return new_Audio_Tensor

def WaveNet(audio_Tensor, mel_Tensor, name= None, reuse= None):
    with tf.variable_scope(name, 'wavenet', reuse= reuse):
        input_Channels = audio_Tensor.get_shape()[-1]

        output_Tensor = 0;

        audio_Tensor = Weight_Norm_Conv1D(
            inputs= audio_Tensor,
            filters= hp.WaveGlow.WaveNet.Channels,
            kernel_size= 1,
            strides= 1,
            padding='same',
            name= 'audio_initial_conv'
            )

        for index in range(hp.WaveGlow.WaveNet.Layers):
            in_Tensor = Weight_Norm_Conv1D(
                inputs= audio_Tensor,
                filters= hp.WaveGlow.WaveNet.Channels * 2,
                kernel_size= hp.WaveGlow.WaveNet.Kernel_Size,
                strides= 1,
                padding='same',
                dilations= 2 ** index,
                name= 'audio_in_{}'.format(index)
                )
            cond_Tensor = Weight_Norm_Conv1D(
                inputs= mel_Tensor,
                filters= hp.WaveGlow.WaveNet.Channels * 2,
                kernel_size= 1,
                strides= 1,
                padding='same',
                dilations= 2 ** index,
                name= 'mel_cond_{}'.format(index)
                )

            audio_Tensor = tf.split(
                in_Tensor + cond_Tensor,
                num_or_size_splits= 2,
                axis=-1
                )
            audio_Tensor = tf.tanh(audio_Tensor[0]) * tf.sigmoid(audio_Tensor[1]);

            res_Skip_Tensor = Weight_Norm_Conv1D(
                inputs= audio_Tensor,
                filters= hp.WaveGlow.WaveNet.Channels * 2 if index < hp.WaveGlow.WaveNet.Layers - 1 else hp.WaveGlow.WaveNet.Channels,
                kernel_size= 1,
                strides= 1,
                padding='same',
                name= 'res_{}'.format(index)
                )

            if index < hp.WaveGlow.WaveNet.Layers - 1:
                res_Audio_Tensor, skip_Activation_Tensor = tf.split(
                    res_Skip_Tensor,
                    num_or_size_splits= 2,
                    axis=-1
                    )
                audio_Tensor += res_Audio_Tensor
            else:
                skip_Activation_Tensor = res_Skip_Tensor

            output_Tensor += skip_Activation_Tensor

        output_Tensor = tf.layers.conv1d(
            output_Tensor,
            filters= input_Channels * 2,
            kernel_size= 1,
            kernel_initializer= tf.initializers.zeros(),
            bias_initializer= tf.initializers.zeros()
            )

        return tf.split(
            output_Tensor,
            num_or_size_splits= 2,
            axis=-1
            )

def Glow_Train(audio_Tensor, mel_Tensor):
    output_Audio_List = [];
    log_S_List = [];
    log_Det_W_List = [];
    for flow_Index in range(hp.WaveGlow.Flows):
        if flow_Index % hp.WaveGlow.Early_Every == 0 and flow_Index > 0:
            output_Audio_List.append(audio_Tensor[:, :, :hp.WaveGlow.Early_Size])
            audio_Tensor = audio_Tensor[:, :, hp.WaveGlow.Early_Size:]

        audio_Tensor, log_S_Tensor, log_Det_W_Tensor = Affine_Coupling_Layer(
            audio_Tensor= audio_Tensor,
            mel_Tensor= mel_Tensor,
            reverse= False,
            name= 'affine_coupling_layer_{}'.format(flow_Index)
            )
        
        log_S_List.append(log_S_Tensor);
        log_Det_W_List.append(log_Det_W_Tensor);

    output_Audio_List.append(audio_Tensor)

    output_Audio_Tensor = tf.concat(output_Audio_List, axis=-1)

    return output_Audio_Tensor, log_S_List, log_Det_W_List

def Glow_Inference(audio_Tensor, mel_Tensor, sigma=1.0):
    for flow_Index in reversed(range(hp.WaveGlow.Flows)):
        audio_Tensor = Affine_Coupling_Layer(
            audio_Tensor= audio_Tensor,
            mel_Tensor= mel_Tensor,
            reverse= True,
            name= 'affine_coupling_layer_{}'.format(flow_Index)
            )

        if flow_Index % hp.WaveGlow.Early_Every == 0 and flow_Index > 0:
            z = tf.random.normal((
                tf.shape(mel_Tensor)[0],
                tf.shape(mel_Tensor)[1],
                hp.WaveGlow.Early_Size
                ))
            audio_Tensor = tf.concat([z * sigma, audio_Tensor], axis=-1)
                
    return tf.reshape(audio_Tensor, shape=(tf.shape(audio_Tensor)[0], -1))

def Glow_Loss(audios_Labels, output_Audio_Tensor, log_S_List, log_Det_W_List, sigma=1.0):
    audio_Size = tf.cast(tf.size(output_Audio_Tensor), tf.float32)

    log_S_Loss = tf.reduce_sum(log_S_List)
    log_S_Loss = -log_S_Loss / audio_Size

    log_Det_W_Loss = tf.reduce_sum(log_Det_W_List)
    log_Det_W_Loss = -log_Det_W_Loss / audio_Size

    audio_Loss = tf.reduce_sum(output_Audio_Tensor ** 2) / (2 * sigma ** 2)
    audio_Loss = audio_Loss / audio_Size

    #output_Audio_Tensor = tf.reshape(output_Audio_Tensor, shape=(tf.shape(output_Audio_Tensor)[0], -1))

    #audios_Labels = tf.cond(
    #    tf.greater(
    #        tf.shape(audios_Labels)[1],
    #        tf.shape(output_Audio_Tensor)[1]
    #        ),
    #    true_fn= lambda: tf.slice(
    #        audios_Labels,
    #        begin=[0, 0],
    #        size=[tf.shape(audios_Labels)[0], tf.shape(output_Audio_Tensor)[1]]
    #        ),
    #    false_fn= lambda: audios_Labels
    #    )
    #output_Audio_Tensor = tf.cond(
    #    tf.greater(
    #        tf.shape(output_Audio_Tensor)[1],
    #        tf.shape(audios_Labels)[1]
    #        ),
    #    true_fn= lambda: tf.slice(
    #        output_Audio_Tensor,
    #        begin=[0, 0],
    #        size=[tf.shape(output_Audio_Tensor)[0], tf.shape(audios_Labels)[1]]
    #        ),
    #    false_fn= lambda: output_Audio_Tensor
    #    )
    #abs_Loss = tf.losses.absolute_difference(labels= audios_Labels, predictions= output_Audio_Tensor)  #Test....
    
    
    return log_S_Loss, log_Det_W_Loss, audio_Loss#, abs_Loss

def Reshaped_Mel(mel_Tensor):
    batch_Size = tf.shape(mel_Tensor)[0]

    padding_Step = tf.mod(
        hp.WaveGlow.Inference.Mel_Split_Length - tf.mod(tf.shape(mel_Tensor)[1], hp.WaveGlow.Inference.Mel_Split_Length),
        hp.WaveGlow.Inference.Mel_Split_Length
        )

    mel_Tensor = tf.concat(
        [mel_Tensor, tf.zeros((batch_Size, padding_Step, hp.Sound.Mel_Dim))],
        axis= 1
        )

    mel_Tensor = tf.reshape(
        mel_Tensor,
        (
            batch_Size * (tf.shape(mel_Tensor)[1] // hp.WaveGlow.Inference.Mel_Split_Length),
            hp.WaveGlow.Inference.Mel_Split_Length,
            hp.Sound.Mel_Dim
            )
        )

    return mel_Tensor