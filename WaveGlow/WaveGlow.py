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
from scipy.io.wavfile import write
from WaveGlow import Modules, Feeder;
import Hyper_Parameters as hp;


class WaveGlow:
    def __init__(self):
        self.tf_Session = tf.Session();

        self.feeder = Feeder.Feeder();

        self.Tensor_Generate();

        self.tf_Saver = tf.train.Saver(max_to_keep= 5);

    def Tensor_Generate(self):
        global_Step = tf.train.get_or_create_global_step()
        
        print('Training tensor generate...')
        self.Train_Tensor_Generate();
        print('Inference tensor generate...')
        self.Inference_Tensor_Generate();
        print('Variables initializing...')
        self.tf_Session.run(tf.global_variables_initializer());
        print('Tensor generating done.')
            
    def Train_Tensor_Generate(self):
        placeholder_Dict = self.feeder.placeholder_Dict;

        with tf.variable_scope('waveglow') as scope:
            audio_Tensor, mel_Tensor = Modules.Restructure_Train_Data(
                audios= placeholder_Dict['Audio'],
                mels= placeholder_Dict['Mel']
                )   #Audio: [N, T/G, G], Mel: [N, T/G, G*C]

            output_Audio_Tensor, log_S_List, log_Det_W_List = Modules.Glow_Train(audio_Tensor, mel_Tensor)
            
        with tf.variable_scope('loss') as scope:
            log_S_Loss, log_Det_W_Loss, audio_Loss = Modules.Glow_Loss(placeholder_Dict['Audio'], output_Audio_Tensor, log_S_List, log_Det_W_List)
            #log_S_Loss, log_Det_W_Loss, audio_Loss, abs_Loss = Modules.Glow_Loss(placeholder_Dict['Audio'], output_Audio_Tensor, log_S_List, log_Det_W_List)
            global_Step = tf.train.get_or_create_global_step()

            learning_Rate = tf.train.exponential_decay(
                learning_rate= hp.WaveGlow.Train.Learning_Rate.Initial,
                global_step= global_Step,
                decay_steps= hp.WaveGlow.Train.Learning_Rate.Decay_Step,
                decay_rate= hp.WaveGlow.Train.Learning_Rate.Decay_Rate,
                )
            learning_Rate = tf.maximum(hp.WaveGlow.Train.Learning_Rate.Min, learning_Rate)

            optimizer = tf.train.AdamOptimizer(
                learning_rate= learning_Rate,
                beta1= hp.WaveGlow.Train.ADAM.Beta1,
                beta2= hp.WaveGlow.Train.ADAM.Beta2,
                epsilon= hp.WaveGlow.Train.ADAM.Epsilon,
                )
            gradients, variables = zip(*optimizer.compute_gradients(log_S_Loss + log_Det_W_Loss + audio_Loss))
            #gradients, variables = zip(*optimizer.compute_gradients(log_S_Loss + log_Det_W_Loss + audio_Loss + abs_Loss))
            clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 0.1)
            train_Op = tf.group([
                tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                optimizer.apply_gradients(zip(clipped_Gradients, variables), global_step=global_Step)
                ])

        self.train_Tensor_Dict = {
            'Global_Step': global_Step,
            'Learning_Rate': learning_Rate,
            'Log_S_Loss': log_S_Loss,
            'Log_Det_W_Loss': log_Det_W_Loss,
            'Audio_Loss': audio_Loss,
            #'ABS_Loss': abs_Loss,
            'Train_OP': train_Op,
            }

    def Inference_Tensor_Generate(self):
        placeholder_Dict = self.feeder.placeholder_Dict;

        with tf.variable_scope('waveglow', reuse=tf.AUTO_REUSE) as scope:
            audio_Tensor, mel_Tensor = Modules.Restructure_Inference_Data(
                mels= placeholder_Dict['Mel']
                )

            output_Audio_Tensor = Modules.Glow_Inference(audio_Tensor, mel_Tensor)
        
        self.inference_Tensor_Dict = {
            'Global_Step': tf.train.get_or_create_global_step(),
            'Audio': output_Audio_Tensor
            }
        
    def Restore(self):
        latest_Checkpoint = tf.train.latest_checkpoint(hp.WaveGlow.Checkpoint_Path);
        if latest_Checkpoint is None:
            print('There is no checkpoint.');
            return;

        self.tf_Saver.restore(self.tf_Session, latest_Checkpoint);
        print('Checkpoint \'{}\' is loaded.'.format(latest_Checkpoint));

    def Train(self):
        def Run_Inference():
            with open('WaveGlow_Inference_File_Path_in_Train.txt', 'r') as f:
                path_List = [path.strip() for path in f.readlines()]
            self.Inference(path_List)

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
                'Log S Loss: {:0.5f}'.format(result_Dict['Log_S_Loss']),
                'Log Det W Loss: {:0.5f}'.format(result_Dict['Log_Det_W_Loss']),
                'Audio Loss: {:0.5f}'.format(result_Dict['Audio_Loss']),
                #'ABS Loss: {:0.5f}'.format(result_Dict['ABS_Loss']),
                ]
            print('\t\t'.join(display_List))

            #global step의 update 타이밍이 좀 바뀐거 같기도 하고??????
            if (result_Dict['Global_Step'] + 1) % hp.WaveGlow.Train.Checkpoint_Save_Timing == 0:
                os.makedirs(os.path.join(hp.WaveGlow.Checkpoint_Path).replace("\\", "/"), exist_ok= True);
                self.tf_Saver.save(self.tf_Session, os.path.join(hp.WaveGlow.Checkpoint_Path, 'WG_CHECKPOINT').replace('\\', '/'), global_step= result_Dict['Global_Step'] + 1);
            if (result_Dict['Global_Step'] + 1) % hp.WaveGlow.Train.Inference_Timing == 0:
                Run_Inference();

    def Inference(self, path_List, file_Prefix= None):
        os.makedirs(os.path.join(hp.WaveGlow.Inference.Path, 'WAV').replace("\\", "/"), exist_ok= True);
        os.makedirs(os.path.join(hp.WaveGlow.Inference.Path, 'PLOT').replace("\\", "/"), exist_ok= True);

        original_Sig_List, feed_Dict_List, mel_Index_List = self.feeder.Get_Inference_Pattern(path_List)
        result_Dict_List = []
        for feed_Dict in feed_Dict_List:
            result_Dict_List.append(self.tf_Session.run(
                fetches= self.inference_Tensor_Dict,
                feed_dict= feed_Dict
                ))

        result_Sig = np.zeros(
            shape= [                
                sum([result_Dict['Audio'].shape[0] for result_Dict in result_Dict_List]),
                max([result_Dict['Audio'].shape[1] for result_Dict in result_Dict_List])
                ],
            dtype= np.float32
            )
        current_Index = 0;
        for result_Dict in result_Dict_List:
            result_Sig[current_Index:current_Index + result_Dict['Audio'].shape[0], :result_Dict['Audio'].shape[1]] = result_Dict['Audio']
            current_Index += result_Dict['Audio'].shape[0]

        result_Sig_List = [np.reshape(result_Sig[start_Index:end_Index], [-1]) for start_Index, end_Index in mel_Index_List]
                
        export_Inference_Thread = Thread(
            target= self.Export_Inference,
            args= [
                path_List,
                original_Sig_List,
                result_Sig_List,
                result_Dict['Global_Step'],
                file_Prefix or 'GS_{}'.format(result_Dict['Global_Step'])
                ]
            )
        export_Inference_Thread.daemon = True;
        export_Inference_Thread.start();

    def Export_Inference(self, path_List, original_Sig_List, result_Sig_List, global_Step, file_Prefix='Inference'):
        for index, (path, original_Sig, result_Sig) in enumerate(zip(path_List, original_Sig_List, result_Sig_List)):
            result_Sig =  librosa.util.normalize(result_Sig)
            write(
                os.path.join(hp.WaveGlow.Inference.Path, 'WAV', '{}.IDX_{}.WAV'.format(file_Prefix, index)).replace("\\", "/"),
                hp.WaveGlow.Export_Sample_Rate,
                np.floor(np.clip(result_Sig, -1, 1) * 32768).astype(np.int16)
                )

            new_Figure = plt.figure(figsize=(32, 9), dpi=100);
            plt.subplot(2,1,1);        
            plt.plot(original_Sig)
            plt.title('Global step: {}    Wav file: {}    Original wav'.format(global_Step, path))
            plt.subplot(2,1,2);
            plt.plot(result_Sig)
            plt.title('Global step: {}    Wav file: {}    Result wav'.format(global_Step, path))
            plt.tight_layout()
            plt.savefig(
                os.path.join(hp.WaveGlow.Inference.Path, 'PLOT', '{}.IDX_{}.PNG'.format(file_Prefix, index)).replace("\\", "/"),
                )
            plt.close(new_Figure);
        
if __name__ == '__main__':
    new_WaveGlow = WaveGlow();
    new_WaveGlow.Restore();
    new_WaveGlow.Train();