import tensorflow as tf;
import Hyper_Parameters as hp;
from ZoneoutLSTMCell import ZoneoutLSTMCell
from Audio import inv_spectrogram

#from tensorflow.python.ops import rnn, control_flow_util, variable_scope

def ConvBank(inputs, is_training= False):
    new_Tensor = inputs;
    for index in range(hp.Taco1_Mel_to_Spect.ConvBank.Nums):
        with tf.variable_scope('convbank_{}'.format(index)):
            new_Tensor_List = [
                tf.layers.conv1d(
                    inputs= new_Tensor,
                    filters= hp.Taco1_Mel_to_Spect.ConvBank.Channel,
                    kernel_size= kernel_Size + 1,
                    strides= hp.Taco1_Mel_to_Spect.ConvBank.Stride,
                    padding= 'same',
                    activation= tf.nn.relu
                    )
                for kernel_Size in range(hp.Taco1_Mel_to_Spect.ConvBank.Max_Kernel_Size)
                ]
            new_Tensor_List = [
                tf.layers.batch_normalization(x, training= is_training)
                for x in new_Tensor_List
                ]
            new_Tensor = tf.concat(new_Tensor_List, axis=-1);
            new_Tensor = tf.layers.max_pooling1d(
                inputs= new_Tensor,
                pool_size= hp.Taco1_Mel_to_Spect.ConvBank.Pooling.Size,
                strides= hp.Taco1_Mel_to_Spect.ConvBank.Pooling.Stride,
                padding= 'same'
                )
            new_Tensor = tf.layers.conv1d(
                inputs= new_Tensor,
                filters= hp.Taco1_Mel_to_Spect.ConvBank.Projection1.Channel,
                kernel_size= hp.Taco1_Mel_to_Spect.ConvBank.Projection1.Kernel_Size,
                strides= hp.Taco1_Mel_to_Spect.ConvBank.Projection1.Stride,
                padding= 'same',
                activation= tf.nn.relu
                )
            new_Tensor = tf.layers.batch_normalization(new_Tensor, training= is_training)
            new_Tensor = tf.layers.conv1d(
                inputs= new_Tensor,
                filters= hp.Taco1_Mel_to_Spect.ConvBank.Projection2.Channel,
                kernel_size= hp.Taco1_Mel_to_Spect.ConvBank.Projection2.Kernel_Size,
                strides= hp.Taco1_Mel_to_Spect.ConvBank.Projection2.Stride,
                padding= 'same'
                )
            new_Tensor = tf.layers.batch_normalization(new_Tensor, training= is_training)

    return inputs + new_Tensor

def Highway(inputs):
    highway_Size = inputs.get_shape()[-1];
    new_Tensor = inputs;
    for index in range(hp.Taco1_Mel_to_Spect.Highway.Nums):
        with tf.variable_scope('highway_{}'.format(index)):
            H = tf.layers.dense(
                inputs= new_Tensor,
                units= highway_Size, #hp.Taco1_Mel_to_Spect.Highway.Size,
                activation= tf.nn.relu
                )
            T = tf.layers.dense(
                inputs= new_Tensor,
                units= highway_Size, #hp.Taco1_Mel_to_Spect.Highway.Size,
                activation= tf.nn.sigmoid,
                bias_initializer= tf.constant_initializer(-1.0)
                )
            new_Tensor = H*T + new_Tensor * (1.0 - T)

    return new_Tensor
            
#귀찮으니 그냥 ZoneoutLSTM으로 간다!
def BiRNN(inputs, is_training = False):
    cell_List_Dict = {
        'Forward': [],
        'Backward': []
        }
    for index in range(hp.Taco1_Mel_to_Spect.BiRNN.Nums):
        for direction in ['Forward', 'Backward']:
            with tf.variable_scope('birnncell_{}_{}'.format(index, direction.lower())):
                cell_List_Dict[direction].append(ZoneoutLSTMCell(
                    num_units= hp.Taco1_Mel_to_Spect.BiRNN.Cell_Size,
                    is_training= is_training,
                    cell_zoneout_rate= hp.Taco1_Mel_to_Spect.BiRNN.Zoneout_Rate,
                    output_zoneout_rate= hp.Taco1_Mel_to_Spect.BiRNN.Zoneout_Rate
                    ))

    with tf.variable_scope('birnn'):
        new_Tensor, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw= cell_List_Dict['Forward'],
            cells_bw= cell_List_Dict['Backward'],
            inputs= inputs,
            dtype= tf.float32,
            sequence_length= None,        
            )

    return new_Tensor

def Projection(inputs):
    return tf.layers.dense(
        inputs,
        hp.Sound.Spectrogram_Dim
        )

def Loss(x, y):
    return tf.losses.absolute_difference(x, y);

def Griffin_Lim(spectrogram):
    #spectrogram: [Time, Dim]
    return inv_spectrogram(
        spectrogram= spectrogram.transpose(), #[Dim, Time]
        num_freq= hp.Sound.Spectrogram_Dim,
        frame_shift_ms= hp.Sound.Frame_Shift,
        frame_length_ms= hp.Sound.Frame_Length,
        sample_rate= hp.Sound.Sample_Rate,
        griffin_lim_iters= hp.Taco1_Mel_to_Spect.Griffin_Lim_Iteration
        )