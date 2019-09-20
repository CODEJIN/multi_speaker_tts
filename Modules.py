import tensorflow as tf;
import Hyper_Parameters as hp;
from ZoneoutLSTMCell import ZoneoutLSTMCell
from Location_Sensitive_Attention import Location_Sensitive_Attention;
from tensorflow.contrib.seq2seq import Helper, AttentionWrapper;
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn, control_flow_util, variable_scope
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder
from tensorflow.contrib.seq2seq.python.ops.decoder import Decoder
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_util
from collections import namedtuple

def Encoder_Embedding(inputs):
    embedding_Variable = tf.get_variable(
        name = "embedding_variable",
        shape = [hp.Encoder.Embedding.Token_Size, hp.Encoder.Embedding.Embedding_Size],
        dtype = tf.float32
        )
    new_Tensor = tf.nn.embedding_lookup(embedding_Variable, inputs);

    return new_Tensor;

def Encoder_Conv(inputs, is_training = False):
    new_Tensor = inputs;
    for index in range(hp.Encoder.Conv.Nums):
        with tf.variable_scope('conv_{}'.format(index)):
            new_Tensor = tf.layers.conv1d(
                inputs= new_Tensor,
                filters= hp.Encoder.Conv.Channel,
                kernel_size= hp.Encoder.Conv.Kernel_Size,
                strides= hp.Encoder.Conv.Stride,
                padding= 'same',
                activation= tf.nn.relu
                )
            new_Tensor = tf.layers.batch_normalization(
                new_Tensor,
                training= is_training
                )
            new_Tensor = tf.layers.dropout(
                new_Tensor,
                rate= hp.Encoder.Conv.Dropout_Rate,
                training= is_training
                )

    return new_Tensor;

def Encoder_BiLSTM(inputs, lengths, is_training = False):
    cell_List_Dict = {
        'Forward': [],
        'Backward': []
        }
    for index in range(hp.Encoder.BiLSTM.Nums):
        for direction in ['Forward', 'Backward']:
            with tf.variable_scope('bilstmcell_{}_{}'.format(index, direction.lower())):
                cell_List_Dict[direction].append(ZoneoutLSTMCell(
                    num_units= hp.Encoder.BiLSTM.Cell_Size,
                    is_training= is_training,
                    cell_zoneout_rate= hp.Encoder.BiLSTM.Zoneout_Rate,
                    output_zoneout_rate= hp.Encoder.BiLSTM.Zoneout_Rate
                    ))

    with tf.variable_scope('bilstm'):
        new_Tensor, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw= cell_List_Dict['Forward'],
            cells_bw= cell_List_Dict['Backward'],
            inputs= inputs,
            dtype= tf.float32,
            sequence_length= lengths,        
            )

    return new_Tensor


def Decoder_LSTM(inputs, sequence_length, attention_mechanism, is_training= False):
    '''
    In inference, input and sequence_length will be ignoired.
    '''
    cell_List = [];
    for index in range(hp.Decoder.LSTM.Nums):
        cell_List.append(ZoneoutLSTMCell(
            num_units= hp.Decoder.LSTM.Cell_Size,
            is_training= is_training,
            cell_zoneout_rate= hp.Decoder.LSTM.Zoneout_Rate,
            output_zoneout_rate= hp.Decoder.LSTM.Zoneout_Rate
            ))
    lstm_Cell = tf.nn.rnn_cell.MultiRNNCell(cell_List);
    
    attention_Wrapped_Cell = AttentionWrapper(
        cell= lstm_Cell,
        attention_mechanism= attention_mechanism,
        attention_layer_size=None,
        alignment_history=True,
        cell_input_fn=None,
        output_attention= False,
        initial_cell_state=None,
        name=None,
        attention_layer=None
        )

    helper = Decoder_Helper(
        inputs= inputs, #Mel
        sequence_length= sequence_length,   #Mel_length
        time_major= False,
        is_training= is_training,
        name= None
        )
    decoder = Decoder_Decoder(
        cell= attention_Wrapped_Cell,
        helper= helper,
        initial_state= attention_Wrapped_Cell.zero_state(tf.shape(inputs)[0], tf.float32)
        )
    final_outputs, final_state, _ = Decoder_Dynamic_Decode(
        decoder= decoder,
        impute_finished= False  #True
        )

    return final_outputs, final_state

def Decoder_Conv(inputs, is_training = False):
    new_Tensor = inputs;
    for index in range(hp.Decoder.Conv.Nums):
        with tf.variable_scope('conv_{}'.format(index)):
            new_Tensor = tf.layers.conv1d(
                inputs= new_Tensor,
                filters= hp.Decoder.Conv.Channel if index < hp.Decoder.Conv.Nums - 1 else hp.Sound.Mel_Dim,
                kernel_size= hp.Decoder.Conv.Kernel_Size,
                strides= hp.Decoder.Conv.Stride,
                padding= 'same',
                activation= tf.nn.tanh
                )
            new_Tensor = tf.layers.batch_normalization(
                new_Tensor,
                training= is_training
                )
            new_Tensor = tf.layers.dropout(
                new_Tensor,
                rate= hp.Encoder.Conv.Dropout_Rate,
                training= is_training
                )

    return new_Tensor;


#nest.map_structure: 입력을 tensor들의 list나 dict으로 받고, 모든 element에 대해 function을 수행한 후 같은 구조체 형태로 반환
#TensorArray.read(x): Dim 0의 x번째를 indexing함(ex: array[5,3,7]일때 x=3이라면, array[3,:,:]를 반환)
class Decoder_Helper(Helper):
    def __init__(
        self,
        inputs,
        sequence_length,
        time_major= False,
        is_training= False,
        name= None
        ):
        self._inputs = inputs;
        self._sequence_length = tf.convert_to_tensor(sequence_length, name="sequence_length")
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError(
                "Expected sequence_length to be a vector, but received shape: %s" %
                self._sequence_length.get_shape()
                )
        self.time_major = time_major;
        self.is_training = is_training;
        self._batch_size = tf.shape(self._inputs)[0];
        
        if not time_major:
            inputs = rnn._transpose_batch_time(inputs);        
        self._zero_inputs = tf.zeros_like(inputs[0, :])

        self._input_tas = tf.TensorArray(
            dtype=inputs.dtype,
            size=tf.shape(inputs)[0],
            element_shape=inputs.get_shape()[1:]
            ).unstack(inputs)

    def initialize(self, name= None):
        initial_finished = tf.tile([False], [self._batch_size])
        initial_inputs = tf.zeros([self._batch_size, hp.Sound.Mel_Dim])  #A zero array is inserted at first time.
        prenet_vector = self.prenet(initial_inputs)
        context_vector = tf.zeros(shape=[self.batch_size, hp.Encoder.BiLSTM.Cell_Size * 2 + hp.Speaker_Embedding.Embedding_Size])
        initial_inputs = tf.concat([prenet_vector, context_vector], axis=-1)

        return (initial_finished, initial_inputs)

    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with tf.name_scope(name, "HelperSample", [time, outputs]):
            sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, logits, stop_logits, state, name= None):
        next_finished = tf.cond(
            self.is_training,
            true_fn= lambda: tf.greater_equal(time, self._sequence_length),
            false_fn=lambda: tf.logical_or(
                tf.squeeze(tf.greater_equal(stop_logits, 0.0), axis=-1),
                tf.greater_equal(time, hp.Decoder.LSTM.Max_Inference_Length)
                )
            )

        next_inputs = tf.cond(
            self.is_training,
            true_fn= lambda: tf.cond(
                tf.reduce_all(next_finished),
                true_fn=lambda: self._zero_inputs,
                false_fn=lambda: self._input_tas.read(time)
                ),
            false_fn= lambda: logits
            )

        prenet_vector = self.prenet(next_inputs)
        context_vector = state.attention
        next_inputs = tf.concat([prenet_vector, context_vector], axis=-1)
        next_state = state

        return (next_finished, next_inputs, next_state)

    def prenet(self, inputs):
        new_Tensor = inputs
        for index in range(hp.Decoder.PreNet.Nums):        
            with tf.variable_scope('prenet_{}'.format(index), reuse=variable_scope.AUTO_REUSE):
                new_Tensor = tf.layers.dense(
                    inputs= new_Tensor,
                    units= hp.Decoder.PreNet.Size,
                    activation= tf.nn.relu
                    )
                if hp.Decoder.PreNet.Use_Dropout:
                    new_Tensor = tf.layers.dropout(
                        inputs= new_Tensor,
                        rate= hp.Decoder.PreNet.Dropout_Rate,
                        training= True
                        )

        return new_Tensor

class Decoder_Output(namedtuple('Decoder_Output', ('linear', 'stop'))):
    pass

class Decoder_Decoder(BasicDecoder):
    def __init__(
        self,
        cell,
        helper,
        initial_state
        ):
        super(Decoder_Decoder, self).__init__(
            cell=cell,
            helper=helper,
            initial_state=initial_state
            )

    @property
    def output_size(self):
        return Decoder_Output(
            linear= TensorShape([hp.Sound.Mel_Dim]),
            stop= TensorShape([1])  #Current, it is hard code
            )
    @property
    def output_dtype(self):
        return Decoder_Output(
            linear= tf.float32,
            stop= tf.float32
            )

    def step(self, time, inputs, state, name= None):
        with tf.name_scope(name, "DecoderStep", (time, inputs, state)):
            batch_size = tf.shape(inputs)[0]

            cell_outputs, cell_state = self._cell(inputs, state);
            
            context = cell_state.attention
            cell_outputs = tf.concat([cell_outputs, context], axis=-1)
            logits, stop_logits = self.projection(cell_outputs)
            finished, next_inputs, next_state = self._helper.next_inputs(
                time= time,
                logits= logits,
                stop_logits= stop_logits,
                state= cell_state
                )

            outputs = Decoder_Output(
                linear= logits,
                stop= stop_logits
                )

            return outputs, next_state, next_inputs, finished

    def projection(self, inputs):        
        with tf.variable_scope('linear_projection', reuse=variable_scope.AUTO_REUSE):
            new_Tensor = tf.layers.dense(
                inputs= inputs,
                units= hp.Sound.Mel_Dim + 1
                )
            mel_Logits, stop_Logits = tf.split(
                new_Tensor,
                num_or_size_splits=[hp.Sound.Mel_Dim, 1],
                axis=-1
                )

        return mel_Logits, stop_Logits

def Decoder_Dynamic_Decode(
    decoder,
    output_time_major= False,
    impute_finished= False,
    maximum_iterations= None,
    parallel_iterations= 32,
    swap_memory= False,
    scope= None
    ):
    if not isinstance(decoder, Decoder):
        raise TypeError("Expected decoder to be type Decoder, but saw: %s" % type(decoder))

    with variable_scope.variable_scope(scope, "decoder") as varscope:
        ctxt = tf.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
        is_xla = control_flow_util.GetContainingXLAContext(ctxt) is not None
        in_while_loop = control_flow_util.GetContainingWhileContext(ctxt) is not None

        if not context.executing_eagerly() and not in_while_loop:
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = tf.convert_to_tensor(
                maximum_iterations,
                dtype=tf.int32,
                name="maximum_iterations"
                )
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")
        elif is_xla:
            raise ValueError("maximum_iterations is required for XLA compilation.")

        initial_finished, initial_inputs, initial_state = decoder.initialize()
        if maximum_iterations is not None:
            initial_finished = tf.logical_or(
                initial_finished,
                0 >= maximum_iterations
                )
        initial_sequence_lengths = tf.zeros_like(initial_finished, dtype=tf.int32)
        initial_time = tf.constant(0, dtype=tf.int32)


        def _shape(batch_size, from_shape):
            if (not isinstance(from_shape, TensorShape) or from_shape.ndims == 0):
                return TensorShape(None)
            else:
                batch_size = tensor_util.constant_value(tf.convert_to_tensor(batch_size, name="batch_size"))
            return TensorShape([batch_size]).concatenate(from_shape)

        dynamic_size = maximum_iterations is None or not is_xla
        def _create_ta(s, d):
            return tf.TensorArray(
                dtype=d,
                size= 0 if dynamic_size else maximum_iterations,
                dynamic_size= dynamic_size,
                element_shape= _shape(decoder.batch_size, s)
                )

        initial_outputs_ta = nest.map_structure(
            _create_ta,
            decoder.output_size,
            decoder.output_dtype
            )

        def condition(
            unused_time,
            unused_outputs_ta,
            unused_state,
            unused_inputs,
            finished,
            unused_sequence_lengths
            ):
            return tf.logical_not(tf.reduce_all(finished))

        def body(
            time,
            outputs_ta,
            state,
            inputs,
            finished,
            sequence_lengths
            ):
            next_outputs, next_state, next_inputs, decoder_finished = decoder.step(time, inputs, state)
            if decoder.tracks_own_finished:
                next_finished = decoder_finished
            else:
                next_finished = tf.logical_or(decoder_finished, finished)   
                next_finished = tf.reshape(next_finished, [-1]) #reshape이유 1: helper에서 cond에 들어가면 merge가 됨, 2: inference시에 2차원 값이 나옴

                
            next_sequence_lengths = tf.where(
                tf.logical_not(finished),
                x= tf.fill(tf.shape(sequence_lengths), time + 1),
                y= sequence_lengths
                )

            nest.assert_same_structure(state, next_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            if impute_finished:
                new_linear = nest.map_structure(
                    lambda out, zero: tf.where(finished, zero, out),
                    next_outputs.linear,
                    tf.zeros_like(next_outputs.linear)
                    )
                next_outputs._replace(linear= new_linear)

                def _maybe_copy_state(new, cur):
                    if isinstance(cur, tf.TensorArray):
                        pass_through = True
                    else:
                        new.set_shape(cur.shape)
                        pass_through = (new.shape.ndims == 0)
                    return new if pass_through else tf.where(finished, cur, new)

                next_state = nest.map_structure(_maybe_copy_state, next_state, state)

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta, next_outputs)

            return time + 1, outputs_ta, next_state, next_inputs, next_finished, next_sequence_lengths

        res = tf.while_loop(
            cond= condition,
            body= body,
            loop_vars=[
                initial_time,
                initial_outputs_ta,
                initial_state,
                initial_inputs,
                initial_finished,
                initial_sequence_lengths
                ],
            parallel_iterations=parallel_iterations,
            maximum_iterations=maximum_iterations,
            swap_memory=swap_memory
            )

        final_outputs_ta, final_state, final_sequence_lengths = res[1], res[2], res[5]
        
        final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)
        try:
            final_outputs, final_state = decoder.finalize(final_outputs, final_state, final_sequence_lengths)
        except NotImplementedError:
            pass

        if not output_time_major:
            final_outputs = nest.map_structure(rnn._transpose_batch_time, final_outputs)

    return final_outputs, final_state, final_sequence_lengths
