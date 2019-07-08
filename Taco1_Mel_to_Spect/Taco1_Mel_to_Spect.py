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
from Taco1_Mel_to_Spect import Feeder, Modules;
import Hyper_Parameters as hp;

class Mel_to_Spect:
    def __init__(self):
        self.tf_Session = tf.Session();

        self.feeder = Feeder.Feeder();

        self.Tensor_Generate();

        self.tf_Saver = tf.train.Saver(max_to_keep= 5)

    def Tensor_Generate(self):
        placeholder_Dict = self.feeder.placeholder_Dict;

        with tf.variable_scope('mel_to_spectrogram'):
            spectrogram_Tensor = Modules.ConvBank(
                inputs= placeholder_Dict['Mel'],
                is_training= placeholder_Dict['Is_Training']
                )
            spectrogram_Tensor = Modules.Highway(
                inputs= spectrogram_Tensor
                )
            spectrogram_Tensor = Modules.BiRNN(
                inputs= spectrogram_Tensor,
                is_training= placeholder_Dict['Is_Training']
                )
            spectrogram_Tensor = Modules.Projection(
                inputs= spectrogram_Tensor
                )

        with tf.variable_scope('loss'):
            weight_Regularization_Loss = hp.Taco1_Mel_to_Spect.Train.Weight_Regularization_Rate * tf.reduce_sum([
                tf.nn.l2_loss(variable)
                for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if not (
                    'bias' in variable.name.lower() or
                    'lstm' in variable.name.lower() or
                    'rnn' in variable.name.lower()
                    )
                ])

            loss_List = [
                weight_Regularization_Loss,
                #Modules.Loss(spectrogram_Tensor[:, :-1], placeholder_Dict['Spectrogram'])
                Modules.Loss(spectrogram_Tensor, placeholder_Dict['Spectrogram'])
                ]

            loss_Tensor = tf.reduce_sum(loss_List);

            global_Step = tf.train.get_or_create_global_step()
            learning_Rate = tf.train.exponential_decay(
                learning_rate= hp.Taco1_Mel_to_Spect.Train.Learning_Rate.Initial,
                global_step= global_Step - hp.Taco1_Mel_to_Spect.Train.Learning_Rate.Decay_Start_Step,
                decay_steps= hp.Taco1_Mel_to_Spect.Train.Learning_Rate.Decay_Step,
                decay_rate= hp.Taco1_Mel_to_Spect.Train.Learning_Rate.Decay_Rate,
                )
            learning_Rate = tf.minimum(tf.maximum(learning_Rate, hp.Taco1_Mel_to_Spect.Train.Learning_Rate.Min), hp.Taco1_Mel_to_Spect.Train.Learning_Rate.Initial)

            optimizer = tf.train.AdamOptimizer(
                learning_rate= learning_Rate,
                beta1= hp.Taco1_Mel_to_Spect.Train.ADAM.Beta1,
                beta2= hp.Taco1_Mel_to_Spect.Train.ADAM.Beta2,
                epsilon= hp.Taco1_Mel_to_Spect.Train.ADAM.Epsilon,
                )
            
            train_Op = tf.group([
                tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                optimizer.minimize(loss_Tensor, global_step= global_Step)
                ])

        self.train_Tensor_Dict = {
            'Global_Step': global_Step,
            'Learning_Rate': learning_Rate,
            'Loss': loss_Tensor,
            'Train_OP': train_Op
            }

        self.inference_Tensor_Dict = {
            'Global_Step': global_Step,
            'Mel': placeholder_Dict['Mel'],
            'Spectrogram': spectrogram_Tensor
            }

        self.tf_Session.run(tf.global_variables_initializer());

    def Restore(self):
        latest_Checkpoint = tf.train.latest_checkpoint(hp.Taco1_Mel_to_Spect.Checkpoint_Path);
        if latest_Checkpoint is None:
            print('There is no checkpoint.');
            return;

        self.tf_Saver.restore(self.tf_Session, latest_Checkpoint);
        print('Checkpoint \'{}\' is loaded.'.format(latest_Checkpoint));

    def Train(self):
        def Run_Inference():
            with open('Mel_to_Spect_Inference_in_Train.txt', 'r') as f:
                speaker_Wav_Paths = [line.strip() for line in f.readlines()]
            self.Inference(speaker_Wav_Paths)

        Run_Inference();
        while True:
            start_Time = time.time();
            result_Dict = self.tf_Session.run(
                fetches= self.train_Tensor_Dict,
                feed_dict= self.feeder.Get_Train_Pattern()
                )

            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Global step: {}'.format(result_Dict['Global_Step']),
                'Learning rate: {:0.5f}'.format(result_Dict['Learning_Rate']),
                'Loss: {:0.5f}'.format(result_Dict['Loss']),
                ]
            print('\t\t'.join(display_List))
        
            #global step의 update 타이밍이 좀 바뀐거 같기도 하고??????
            if (result_Dict['Global_Step'] + 1) % hp.Taco1_Mel_to_Spect.Train.Checkpoint_Save_Timing == 0:
                os.makedirs(os.path.join(hp.Taco1_Mel_to_Spect.Checkpoint_Path).replace("\\", "/"), exist_ok= True);
                self.tf_Saver.save(self.tf_Session, os.path.join(hp.Taco1_Mel_to_Spect.Checkpoint_Path, 'CHECKPOINT').replace('\\', '/'), global_step= result_Dict['Global_Step'] + 1);
            if (result_Dict['Global_Step'] + 1) % hp.Taco1_Mel_to_Spect.Train.Inference_Timing == 0:
                Run_Inference();

    def Inference(self, speaker_Wav_Paths):        
        os.makedirs(os.path.join(hp.Taco1_Mel_to_Spect.Train.Inference.Path, 'WAV').replace("\\", "/"), exist_ok= True);
        os.makedirs(os.path.join(hp.Taco1_Mel_to_Spect.Train.Inference.Path, 'PLOT').replace("\\", "/"), exist_ok= True);

        result_Dict_List = [];
        for feed_Dict in self.feeder.Get_Inference_Pattern(speaker_Wav_Paths):
            result_Dict_List.append(self.tf_Session.run(
                fetches= self.inference_Tensor_Dict,
                feed_dict= feed_Dict
                ))

        mel_List = []
        spectrogram_List = []
        for result_Dict in result_Dict_List:
            mel_List.extend(list(result_Dict['Mel']))
            spectrogram_List.extend(list(result_Dict['Spectrogram']))

        export_Inference_Thread = Thread(
            target=self.Export_Inference,
            args=[
                mel_List,
                spectrogram_List,
                'GS_{}'.format(result_Dict_List[0]['Global_Step'])
                ]
            )
        export_Inference_Thread.daemon = True;
        export_Inference_Thread.start();

    def Export_Inference(self, mel_List, spectrogram_List, prefix='Inference'):
        for index, (mel, spect) in enumerate(zip(mel_List, spectrogram_List)):
            file_Name = '{}.IDX_{}'.format(prefix, index)
            if spect.shape[0] == 1:
                print('WAV \'{}\' exporting failed. The exported spectrogram is too short.'.format(file_Name))
                return;
            else:
                try:
                    wav = Modules.Griffin_Lim(spect)
                    librosa.output.write_wav(
                        path= os.path.join(hp.Taco1_Mel_to_Spect.Train.Inference.Path, 'WAV', '{}.WAV'.format(file_Name)).replace("\\", "/"),
                        y= wav,
                        sr=hp.Sound.Sample_Rate
                        )
                except Exception as e:            
                    print('Wav exporting failed: {}'.format(e))

            new_Figure = plt.figure(figsize=(16, 8), dpi=100);
            plt.subplot(2,1,1);
            plt.imshow(np.transpose(mel), aspect='auto', origin='lower')
            plt.title('Mel')
            plt.colorbar()
            plt.subplot(2,1,2);
            plt.imshow(np.transpose(spect), aspect='auto', origin='lower')
            plt.title('Spectrogram')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(
                os.path.join(hp.Taco1_Mel_to_Spect.Train.Inference.Path, 'PLOT', '{}.PNG'.format(file_Name)).replace("\\", "/"),
                #bbox_inches='tight'
                )
            plt.close(new_Figure);
            
if __name__ == '__main__':
    new_Mel_to_Spect = Mel_to_Spect()
    new_Mel_to_Spect.Restore()
    new_Mel_to_Spect.Train()
