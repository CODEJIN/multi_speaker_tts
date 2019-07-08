#Reference: HN lab 'side_seq2seq_location_sensitive_attention_mechanism.py'
#from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.python.layers import convolutional
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
import tensorflow as tf

class Location_Sensitive_Attention(BahdanauAttention):
    def __init__(
        self,
        num_units,
        memory,
        memory_length,
        conv_kernel_size,
        conv_stride_size,
        conv_channel,
        dropout_rate,
        is_training,
        weight_initializer= None,
        bias_initializer= None,
        ):
        self.num_units = num_units
        self.memory = memory
        self.memory_length = memory_length
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride_size = conv_stride_size
        self.conv_channel = conv_channel
        self.dropout_rate = dropout_rate
        self.is_training = is_training
        self.weight_initializer= weight_initializer or tf.contrib.layers.xavier_initializer()
        self.bias_initializer= bias_initializer or tf.zeros_initializer()

        super(Location_Sensitive_Attention, self).__init__(
            num_units= self.num_units,
            memory= self.memory,
            memory_sequence_length= self.memory_length,
            name='BahdanauAttention'
            )

    def __call__(self, query, state):
        with variable_scope.variable_scope(None, 'location_sensitive_attention', [query]):
            key = self._keys;
            query = self.query_layer(query) if self.query_layer else query;
            query = array_ops.expand_dims(query, axis=1);
            proceeding_alignment = array_ops.expand_dims(state, axis=2);            
            with variable_scope.variable_scope('attention_convolution_dense_layer', reuse=variable_scope.AUTO_REUSE):
                proceeding_alignment = convolutional.conv1d(
                    inputs= proceeding_alignment,
                    filters= self.conv_channel,
                    kernel_size= self.conv_kernel_size,
                    strides= self.conv_stride_size,
                    padding= 'same'
                    )
                proceeding_alignment = core.dense(
                    inputs= proceeding_alignment,
                    units= self.num_units,
                    use_bias= False
                    )
            energy = self.score(key, query, proceeding_alignment);

            alignment = self._probability_fn(energy, state)                                                                                     # 4-2. alignment of shape [batch_size, max_encoder_output_lengths]
            cumulated_alignment = alignment + state                                                                                             # 4-3. cumulated_alignment of shape [batch_size, max_encoder_output_lengths]
            next_state = cumulated_alignment                                                                                                    # 4-3. cumulated_alignment of shape [batch_size, max_encoder_output_lengths]
        return alignment, next_state

    def score(self, key, query, proceeding_alignment):
        with variable_scope.variable_scope('score_layer', reuse=variable_scope.AUTO_REUSE):
            weight_w = variable_scope.get_variable(
                name= 'weight_w',
                shape= [1,1, self.num_units],
                initializer= self.weight_initializer
                )
            bias_b = variable_scope.get_variable(
                name= 'bias_b',
                shape= [1,1, self.num_units],
                initializer= self.bias_initializer
                )
            energy = math_ops.reduce_sum(
                weight_w * math_ops.tanh(key + query + proceeding_alignment + bias_b),
                axis=2
                )
        return energy;