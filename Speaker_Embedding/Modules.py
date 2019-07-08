import tensorflow as tf;
from tensorflow.nn.rnn_cell import MultiRNNCell, ResidualWrapper;
from ZoneoutLSTMCell import ZoneoutLSTMCell
import Hyper_Parameters as hp;

def Restructure(inputs):    #for residual
    return tf.layers.dense(
        inputs,
        units= hp.Speaker_Embedding.Embedding_Size
        )

def Stack_LSTM(inputs, lengths, is_training = False):
    cell_List = [];
    for index in range(hp.Speaker_Embedding.LSTM.Nums):
        new_Cell = ZoneoutLSTMCell(
            num_units= hp.Speaker_Embedding.LSTM.Cell_Size,
            num_proj= None if hp.Speaker_Embedding.LSTM.Cell_Size == hp.Speaker_Embedding.Embedding_Size else hp.Speaker_Embedding.Embedding_Size,
            activation= tf.tanh,
            is_training= is_training,
            cell_zoneout_rate= hp.Speaker_Embedding.LSTM.Zoneout_Rate,
            output_zoneout_rate= hp.Speaker_Embedding.LSTM.Zoneout_Rate,
            name= 'lstmcell_{}'.format(index)
            )
        if hp.Speaker_Embedding.LSTM.Use_Residual and index < hp.Speaker_Embedding.LSTM.Nums - 1:
            new_Cell = ResidualWrapper(new_Cell)

        cell_List.append(new_Cell)

    with tf.variable_scope('lstm'):
        new_Tensor, _ = tf.nn.dynamic_rnn(
            cell= MultiRNNCell(cell_List),
            inputs= inputs,
            sequence_length= lengths,
            dtype= tf.float32,
            )

    return new_Tensor;

def Embedding_Generate(inputs):
    return tf.nn.l2_normalize(inputs[:, -1, :], axis=1)

def Loss(embeddings):
    speaker_Size = tf.cast(tf.shape(embeddings)[0] / hp.Speaker_Embedding.Train.Batch_per_Speaker, tf.int32);

    reshaped_Embedding_Tensor = tf.reshape(
        embeddings,
        shape=[
            speaker_Size,
            hp.Speaker_Embedding.Train.Batch_per_Speaker,
            embeddings.get_shape()[-1],
            ]     #[Speaker, Batch_per_Speaker, Embedding]
        )

    sum_Embedding_Tensor = tf.tile(
            tf.reduce_sum(reshaped_Embedding_Tensor, axis=1, keepdims=True),    #[Speaker, 1, Embedding]
            multiples= [1, hp.Speaker_Embedding.Train.Batch_per_Speaker, 1] 
            )
    centroid_for_Within = (sum_Embedding_Tensor - reshaped_Embedding_Tensor) / (hp.Speaker_Embedding.Train.Batch_per_Speaker - 1)  #[Speaker, Batch_per_Speaker, Embedding]
    centroid_for_Between = tf.reduce_mean(reshaped_Embedding_Tensor, axis=1)    #[Speaker, Embedding]

    cosine_Similarity_Weight = tf.Variable(10.0, name='weight', trainable = True);
    cosine_Similarity_Bias = tf.Variable(-5.0, name='bias', trainable = True);

    within_Cosine_Similarity = cosine_Similarity_Weight * Cosine_Similarity(reshaped_Embedding_Tensor, centroid_for_Within) - cosine_Similarity_Bias   #[Speaker, Batch_per_Speaker]

    between_Cosine_Similarity = cosine_Similarity_Weight * Cosine_Similarity2D(embeddings, centroid_for_Between) - cosine_Similarity_Bias,    #[Speaker * Batch_per_Speaker, Speaker]
    between_Cosine_Similarity = tf.reshape(
        between_Cosine_Similarity,
        shape=[
            speaker_Size,
            hp.Speaker_Embedding.Train.Batch_per_Speaker,
            speaker_Size,
            ]
        )     #[Speaker, Batch_per_Speaker, Speaker]
    between_Cosine_Similarity_Filter = 1 - tf.tile(
        tf.expand_dims(tf.eye(speaker_Size), axis=1),
        multiples=[1, hp.Speaker_Embedding.Train.Batch_per_Speaker, 1]
        )  #[Speaker, Batch_per_Speaker, Speaker]
    between_Cosine_Similarity = tf.reshape(
        tf.boolean_mask(between_Cosine_Similarity, between_Cosine_Similarity_Filter),
        shape = (
            speaker_Size,
            hp.Speaker_Embedding.Train.Batch_per_Speaker,
            speaker_Size - 1,
            )
        )   #[speaker, pattern_per_Speaker, Speaker - 1]     Same speaker of first dimension was removed at last dimension.

    if hp.Speaker_Embedding.Train.Loss_Calc_Method.upper() == "Softmax".upper():
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels= tf.zeros(shape=(speaker_Size, hp.Speaker_Embedding.Train.Batch_per_Speaker), dtype=tf.int32),
            logits= tf.concat([tf.expand_dims(within_Cosine_Similarity, axis=-1), between_Cosine_Similarity], axis=-1)
            )
    elif hp.Speaker_Embedding.Train.Loss_Calc_Method.upper() == "Contrast".upper():
        loss = tf.reduce_sum(1 - tf.sigmoid(within_Cosine_Similarity) + tf.reduce_max(between_Cosine_Similarity, axis = -1))
    else:
        raise ValueError('Unsupported loss calc method')

    return loss

def Cosine_Similarity(x,y):
    '''
    Compute the cosine similarity between same row of two tensors.
    Args:
        x: nd tensor (...xMxN).
        y: nd tensor (...xMxN). A tensor of the same shape as x
    Returns:        
        cosine_Similarity: A (n-1)D tensor representing the cosine similarity between the rows. Size is (...xM)
    '''
    return tf.reduce_sum(x * y, axis=-1) / (tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.pow(y, 2), axis=-1)));

def Cosine_Similarity2D(x, y):
    '''
    Compute the cosine similarity between each row of two tensors.
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second dimension.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second dimension.
    Returns:        
        cosine_Similarity: A `Tensor` representing the cosine similarity between the rows. Size is (M x L)
    '''
    tiled_X = tf.tile(tf.expand_dims(x, [1]), multiples = [1, tf.shape(y)[0], 1]);   #[M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [0]), multiples = [tf.shape(x)[0], 1, 1]);   #[M, L, N]
    cosine_Similarity = tf.reduce_sum(tiled_Y * tiled_X, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Y, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_X, 2), axis = 2)) + 1e-8)  #[M, L]
    cosine_Similarity = tf.identity(cosine_Similarity, name="cosine_Similarity");

    return cosine_Similarity;

def Inference(inputs):
    new_Tensor = tf.reshape(
        inputs[:, -1, :],
        shape= [
            tf.shape(inputs)[0] / hp.Speaker_Embedding.Inference.Sample_Nums,   #Batch
            hp.Speaker_Embedding.Inference.Sample_Nums, #Sample_num
            inputs.get_shape()[-1]  #Embedding
            ]
        )
    new_Tensor = tf.reduce_mean(new_Tensor, axis=1)  #[Batch, Embedding]
    return tf.nn.l2_normalize(new_Tensor)


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=(None, None, hp.Speaker_Embedding.Embedding_Size))
    lengths = tf.placeholder(tf.uint8, shape=(None,))
    is_training = tf.placeholder(tf.bool);
    
    r = Stack_LSTM(inputs, lengths, is_training)

    print(Embedding_Generate(r))
    print(Inference(r))
    
    import numpy as np

    tf_Session = tf.Session()
    tf_Session.run(tf.global_variables_initializer());
    a,b = tf_Session.run(
        [Embedding_Generate(r), Inference(r)],
        {
            inputs: np.random.rand(15, 100, 256),
            lengths: np.random.randint(50, 100, size=(15,)),
            is_training: False
            }
        )
    print(a.shape, b.shape)