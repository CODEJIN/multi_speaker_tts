import sys, os, librosa, time;
import tensorflow as tf;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np;
import _pickle as pickle;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
from threading import Thread;
from Location_Sensitive_Attention import Location_Sensitive_Attention
import Modules, Feeder;
import Hyper_Parameters as hp;
#import Tacotron1_Modules;
from Taco1_Mel_to_Spect import Modules as Tacotron1_Modules
from Speaker_Embedding import Modules as Speaker_Embedding_Modules

class Tacotron2:
    def __init__(self, is_Training= False):
        self.is_Training = is_Training

        self.tf_Session = tf.Session();

        self.feeder = Feeder.Feeder(is_Training= is_Training);

        self.Tensor_Generate();

        self.tf_Saver = tf.train.Saver(
            var_list= [
                v for v in tf.all_variables()
                if not (
                    v.name.startswith('speaker_embedding') or
                    v.name.startswith('mel_to_spectrogram')                    
                    )
                ],
            max_to_keep= 5,
            );

        self.Speaker_Embedding_Load()
        self.Vocoder_Load()

    def Tensor_Generate(self):
        placeholder_Dict = self.feeder.placeholder_Dict;
        global_Step = tf.train.get_or_create_global_step()

        with tf.variable_scope('speaker_embedding'):
            embeeding_Tensor = Speaker_Embedding_Modules.Restructure(placeholder_Dict['Speaker_Embedding_Mel']);
            embeeding_Tensor = Speaker_Embedding_Modules.Stack_LSTM(
                inputs= embeeding_Tensor,
                lengths= tf.zeros(tf.shape(embeeding_Tensor)[0:1]) + hp.Speaker_Embedding.Inference.Mel_Frame,
                is_training= placeholder_Dict['Is_Training']
                )
            embeeding_Tensor = Speaker_Embedding_Modules.Inference(embeeding_Tensor)   #[Batch, Speaker_Embedding_Size]            

        with tf.variable_scope('encoder'):
            encoder_Tensor = Modules.Encoder_Embedding(placeholder_Dict['Token']);
            encoder_Tensor = Modules.Encoder_Conv(
                inputs= encoder_Tensor,
                is_training= placeholder_Dict['Is_Training']
                )
            encoder_Tensor = Modules.Encoder_BiLSTM(
                inputs= encoder_Tensor,
                lengths= placeholder_Dict['Token_Length'],
                is_training= placeholder_Dict['Is_Training']
                )

            embeeding_Tensor = tf.tile(tf.expand_dims(embeeding_Tensor, axis= 1), multiples= [1, tf.shape(encoder_Tensor)[1], 1]);  #[Batch, Time, Speaker_Embedding_Size]
            encoder_Tensor = tf.concat([encoder_Tensor, embeeding_Tensor], axis= -1)    #[Batch, Time, Cell_Size * 2 + Speaker_Embedding_Size]

        with tf.variable_scope('attention'):
            attention_Mechanism = Location_Sensitive_Attention(
                num_units= hp.Attention.Memory_Size,
                memory= encoder_Tensor,
                memory_length= placeholder_Dict['Token_Length'],
                conv_kernel_size= hp.Attention.Conv.Kernel_Size,
                conv_stride_size= hp.Attention.Conv.Stride,
                conv_channel= hp.Attention.Conv.Channel,
                dropout_rate= hp.Attention.Conv.Dropout_Rate,
                is_training= placeholder_Dict['Is_Training'],
                )

        with tf.variable_scope('decoder'):
            final_Outputs, final_State = Modules.Decoder_LSTM(
                inputs= placeholder_Dict['Mel'],
                sequence_length= placeholder_Dict['Mel_Length'],
                attention_mechanism= attention_Mechanism,
                is_training= placeholder_Dict['Is_Training']
                )

            postnet_Tensor = Modules.Decoder_Conv(
                inputs= final_Outputs.linear,
                is_training= placeholder_Dict['Is_Training']
                )
            postnet_Tensor = final_Outputs.linear + postnet_Tensor
            attention_History = tf.transpose(final_State.alignment_history.stack(), perm=[1,2,0])
            
        with tf.variable_scope('mel_to_spectrogram'):
            spectrogram_Tensor = Tacotron1_Modules.ConvBank(
                inputs= postnet_Tensor,
                is_training= placeholder_Dict['Is_Training']
                )
            spectrogram_Tensor = Tacotron1_Modules.Highway(
                inputs= spectrogram_Tensor
                )
            spectrogram_Tensor = Tacotron1_Modules.BiRNN(
                inputs= spectrogram_Tensor,
                is_training= placeholder_Dict['Is_Training']
                )
            spectrogram_Tensor = Tacotron1_Modules.Projection(
                inputs= spectrogram_Tensor
                )

        if self.is_Training:
            with tf.variable_scope('loss'):
                stop_Target_Tensor = tf.cast(
                    x= tf.logical_not(tf.sequence_mask(
                        placeholder_Dict['Mel_Length'],
                        maxlen = tf.reduce_max(placeholder_Dict['Mel_Length']) + 1
                        )),
                    dtype= tf.float32
                    )   #Stop은 마지막을 봐야되니까....

                #linear와 postnet은 마지막이 영향을 줄 의미가 없어서....
                linear_Loss = tf.losses.mean_squared_error(final_Outputs.linear[:, :-1], placeholder_Dict['Mel'])
                postnet_Loss = tf.losses.mean_squared_error(postnet_Tensor[:, :-1], placeholder_Dict['Mel'])
                if hp.Train.Use_L1_Loss:
                    linear_Loss += tf.losses.absolute_difference(final_Outputs.linear[:, :-1], placeholder_Dict['Mel'])
                    postnet_Loss += tf.losses.absolute_difference(postnet_Tensor[:, :-1], placeholder_Dict['Mel'])
            
                stop_Loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=stop_Target_Tensor, logits=tf.squeeze(final_Outputs.stop, axis=2)))
                weight_Regularization_Loss = hp.Train.Weight_Regularization_Rate * tf.reduce_sum([
                    tf.nn.l2_loss(variable)
                    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    if not (
                        'bias' in variable.name.lower() or
                        'embedding' in variable.name.lower() or
                        'lstm' in variable.name.lower() or
                        'rnn' in variable.name.lower() or
                        'weight_w' in variable.name.lower() or
                        'projection' in variable.name.lower() or
                        variable.name.startswith('speaker_embedding') or
                        variable.name.startswith('mel_to_spectrogram')
                        )
                    ])
            
                loss_Tensor = tf.reduce_sum([linear_Loss, postnet_Loss, stop_Loss, weight_Regularization_Loss]);

                learning_Rate = tf.train.exponential_decay(
                    learning_rate= hp.Train.Learning_Rate.Initial,
                    global_step= global_Step - hp.Train.Learning_Rate.Decay_Start_Step,
                    decay_steps= hp.Train.Learning_Rate.Decay_Step,
                    decay_rate= hp.Train.Learning_Rate.Decay_Rate,
                    )
                learning_Rate = tf.minimum(tf.maximum(learning_Rate, hp.Train.Learning_Rate.Min), hp.Train.Learning_Rate.Initial)

                optimizer = tf.train.AdamOptimizer(
                    learning_rate= learning_Rate,
                    beta1= hp.Train.ADAM.Beta1,
                    beta2= hp.Train.ADAM.Beta2,
                    epsilon= hp.Train.ADAM.Epsilon,
                    )

                train_Op = tf.group([
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                    optimizer.minimize(
                        loss_Tensor,
                        global_step= global_Step,
                        var_list= [
                            v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                            if not (
                                v.name.startswith('speaker_embedding') or
                                v.name.startswith('mel_to_spectrogram')
                                )
                            ]
                        )
                    ])

            self.train_Tensor_Dict = {
                'Global_Step': global_Step,
                'Learning_Rate': learning_Rate,
                'Loss': loss_Tensor,
                'Linear_Loss': linear_Loss,
                'Postnet_Loss': postnet_Loss,
                'Stop_Loss': stop_Loss,
                'Weight_Regularization_Loss': weight_Regularization_Loss,
                'Train_OP': train_Op,
                }

        self.inference_Tensor_Dict = {
            'Global_Step': global_Step,
            'Linear': final_Outputs.linear,
            'Mel': postnet_Tensor,
            'Stop': tf.sigmoid(tf.squeeze(final_Outputs.stop, axis=2)),
            'Attention_History': attention_History,
            'Spectrogram': spectrogram_Tensor
            }

        self.tf_Session.run(tf.global_variables_initializer());

    def Speaker_Embedding_Load(self):
        speaker_Embedding_Saver = tf.train.Saver(var_list= [v for v in tf.all_variables() if v.name.startswith('speaker_embedding')])
        latest_Checkpoint = tf.train.latest_checkpoint(hp.Speaker_Embedding.Checkpoint_Path)
        if latest_Checkpoint is None:
            raise ValueError('There is no speaker embedding checkpoint!')
        speaker_Embedding_Saver.restore(self.tf_Session, latest_Checkpoint);
        print('Speaker embedding checkpoint \'{}\' is loaded.'.format(latest_Checkpoint));

    def Vocoder_Load(self):        
        Vocoder_Saver = tf.train.Saver(var_list= [v for v in tf.all_variables() if v.name.startswith('mel_to_spectrogram')])
        latest_Checkpoint = tf.train.latest_checkpoint(hp.Taco1_Mel_to_Spect.Checkpoint_Path)
        if latest_Checkpoint is None:
            raise ValueError('There is no vocoder checkpoint!')
        Vocoder_Saver.restore(self.tf_Session, latest_Checkpoint);
        print('Vocoder checkpoint \'{}\' is loaded.'.format(latest_Checkpoint));

    def Restore(self):
        latest_Checkpoint = tf.train.latest_checkpoint(hp.Checkpoint_Path);
        if latest_Checkpoint is None:
            print('There is no checkpoint.');
            return;

        self.tf_Saver.restore(self.tf_Session, latest_Checkpoint);
        print('Checkpoint \'{}\' is loaded.'.format(latest_Checkpoint));

    def Train(self):
        def Run_Inference():
            speaker_Wav_Path_List = []
            sentence_List = []
            with open('Inference_Sentence_in_Train.txt', 'r') as f:
                for line in f.readlines():
                    embedding_Path, sentence = line.strip().split('\t');
                    speaker_Wav_Path_List.append(embedding_Path)
                    sentence_List.append(sentence)
            self.Inference(speaker_Wav_Path_List, sentence_List)

        Run_Inference();

        current_Global_Step = self.tf_Session.run(tf.train.get_or_create_global_step())
        while True:
            start_Time = time.time();
            result_Dict = self.tf_Session.run(
                fetches= self.train_Tensor_Dict,
                feed_dict= self.feeder.Get_Train_Pattern(is_Pre_Train= current_Global_Step < hp.Train.Pre_Step)
                )

            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Global step: {}'.format(result_Dict['Global_Step']),
                'Mode: {}'.format('Pre-train' if current_Global_Step < hp.Train.Pre_Step else 'Main'),
                'Learning rate: {:0.5f}'.format(result_Dict['Learning_Rate']),
                'Linear loss: {:0.5f}'.format(result_Dict['Linear_Loss']),
                'Postnet loss: {:0.5f}'.format(result_Dict['Postnet_Loss']),
                'Stop loss: {:0.5f}'.format(result_Dict['Stop_Loss']),
                'WR loss: {:0.5f}'.format(result_Dict['Weight_Regularization_Loss']),
                ]
            print('\t\t'.join(display_List))
        
            if (result_Dict['Global_Step'] + 1) % hp.Train.Checkpoint_Save_Timing == 0:
                os.makedirs(os.path.join(hp.Checkpoint_Path).replace("\\", "/"), exist_ok= True);
                self.tf_Saver.save(self.tf_Session, os.path.join(hp.Checkpoint_Path, 'CHECKPOINT').replace('\\', '/'), global_step= result_Dict['Global_Step'] + 1);
            if (result_Dict['Global_Step'] + 1) % hp.Train.Inference_Timing == 0:
                Run_Inference();

            current_Global_Step = result_Dict['Global_Step']

    def Inference(self, path_List, text_List, file_Prefix= None):
        os.makedirs(os.path.join(hp.Inference_Path, 'WAV').replace("\\", "/"), exist_ok= True);
        os.makedirs(os.path.join(hp.Inference_Path, 'PLOT').replace("\\", "/"), exist_ok= True);

        result_Dict = self.tf_Session.run(
            fetches= self.inference_Tensor_Dict,
            feed_dict= self.feeder.Get_Inference_Pattern(path_List, text_List)
            )

        export_Inference_Thread = Thread(
            target= self.Export_Inference,
            args= [
                text_List,
                list(result_Dict['Linear']),
                list(result_Dict['Mel']),
                list(result_Dict['Spectrogram']),
                list(result_Dict['Attention_History']),
                list(result_Dict['Stop']),
                'GS_{}'.format(result_Dict['Global_Step']) if file_Prefix is None else file_Prefix
                ]
            )
        export_Inference_Thread.daemon = True;
        export_Inference_Thread.start();

    def Export_Inference(self, text_List, linear_List, mel_List, spectrogram_List, attention_History_List, stop_List, file_Prefix='Inference'):
        for index, (text, linear, mel, spectrogram, attention_History, stop) in enumerate(zip(text_List, linear_List, mel_List, spectrogram_List, attention_History_List, stop_List)):
            file_Name = '{}.IDX_{}'.format(file_Prefix, index)

            slice_Index = np.argmax(stop > 0.5) if any(stop > 0.5) else stop.shape[0]
            linear = linear[:slice_Index]
            mel = mel[:slice_Index]
            stop = stop[:slice_Index]
            attention_History = attention_History[:len(text) + 2, :slice_Index]
            spectrogram = spectrogram[:slice_Index]

            if spectrogram.shape[0] == 1:
                print('WAV \'{}\' exporting failed. The exported spectrogram is too short.'.format(file_Name))
            else:
                try:
                    wav = Tacotron1_Modules.Griffin_Lim(spectrogram)
                    librosa.output.write_wav(
                        path= os.path.join(hp.Inference_Path, 'WAV', '{}.WAV'.format(file_Name)).replace("\\", "/"),
                        y= wav,
                        sr=hp.Sound.Sample_Rate
                        )
                except Exception as e:
                    print('Wav exporting failed: {}'.format(e))

            new_Figure = plt.figure(figsize=(16, 24), dpi=100);
            plt.subplot(5,1,1);
            plt.imshow(np.transpose(linear), aspect='auto', origin='lower')
            plt.title('Text: {}    Linear'.format(text))
            plt.colorbar()
            plt.subplot(5,1,2);
            plt.imshow(np.transpose(mel), aspect='auto', origin='lower')
            plt.title('Text: {}    Mel(Postnet)'.format(text))
            plt.colorbar()
            plt.subplot(5,1,3);
            plt.imshow(np.transpose(spectrogram), aspect='auto', origin='lower')
            plt.title('Text: {}    Spectrogram'.format(text))
            plt.colorbar()
            plt.subplot(5,1,4);
            plt.imshow(np.transpose(attention_History), aspect='auto', origin='lower')
            plt.title('Text: {}    Attention history'.format(text))
            plt.xticks(
                range(attention_History.shape[0]),
                ['<S>'] + list(text) + ['<E>'],
                fontsize = 10
                )
            plt.colorbar()
            plt.subplot(5,1,5);
            plt.plot(stop)
            plt.title('Text: {}    Stop flow'.format(text))
            plt.xlim(0, stop.shape[0])
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(
                os.path.join(hp.Inference_Path, 'PLOT', '{}.PNG'.format(file_Name)).replace("\\", "/"),
                #bbox_inches='tight'
                )
            plt.close(new_Figure);
        
if __name__ == '__main__':
    new_Tacotron2 = Tacotron2(is_Training= True)
    new_Tacotron2.Restore()
    new_Tacotron2.Train()