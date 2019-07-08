import sys, os, librosa, time, argparse;
import tensorflow as tf;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
from threading import Thread;
from sklearn.manifold import TSNE;

from Speaker_Embedding import Modules, Feeder;
import Hyper_Parameters as hp;

class Speaker_Embedding:
    def __init__(self, is_Training = True):
        self.tf_Session = tf.Session();

        self.is_Training = is_Training;
        self.feeder = Feeder.Feeder(is_Training);

        self.Tensor_Generate();
        
        self.tf_Saver = tf.train.Saver(max_to_keep= 5)

    def Tensor_Generate(self):
        placeholder_Dict = self.feeder.placeholder_Dict;
        global_Step = tf.train.get_or_create_global_step()

        with tf.variable_scope('speaker_embedding'):
            encoder_Tensor = Modules.Restructure(placeholder_Dict['Mel']);
            encoder_Tensor = Modules.Stack_LSTM(
                inputs= encoder_Tensor,
                lengths= placeholder_Dict['Mel_Length'],
                is_training= placeholder_Dict['Is_Training']
                )

            embeeding_Tensor = Modules.Embedding_Generate(encoder_Tensor)
            inference_Tensor = Modules.Inference(encoder_Tensor)

        if self.is_Training:
            with tf.variable_scope('loss'):
                loss_Tensor = Modules.Loss(embeeding_Tensor)
                global_Step = tf.train.get_or_create_global_step()
                learning_Rate = tf.train.exponential_decay(
                    learning_rate= hp.Speaker_Embedding.Train.Learning_Rate.Initial,
                    global_step= global_Step,
                    decay_steps= hp.Speaker_Embedding.Train.Learning_Rate.Decay_Step,
                    decay_rate= hp.Speaker_Embedding.Train.Learning_Rate.Decay_Rate,
                    )
                learning_Rate = tf.maximum(learning_Rate, hp.Speaker_Embedding.Train.Learning_Rate.Min)

                optimizer = tf.train.AdamOptimizer(
                    learning_rate= learning_Rate,
                    beta1= hp.Speaker_Embedding.Train.ADAM.Beta1,
                    beta2= hp.Speaker_Embedding.Train.ADAM.Beta2,
                    epsilon= hp.Speaker_Embedding.Train.ADAM.Epsilon,
                    )


                gradients, variables = zip(*optimizer.compute_gradients(loss_Tensor))
                clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 3.0)
                train_Op = tf.group([
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                    optimizer.apply_gradients(zip(clipped_Gradients, variables), global_step=global_Step)
                    ])


                train_Op = tf.group([
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                    optimizer.minimize(loss_Tensor, global_step= global_Step)                
                    ])
            
            self.train_Tensor_Dict = {
                'Global_Step': global_Step,
                'Learning_Rate': learning_Rate,
                'Loss': loss_Tensor,
                'Train_OP': train_Op,
                }

        self.inference_Tensor_Dict = {
            'Global_Step': global_Step,
            'Embedding': inference_Tensor
            }

        self.tf_Session.run(tf.global_variables_initializer());

    def Restore(self):
        latest_Checkpoint = tf.train.latest_checkpoint(hp.Speaker_Embedding.Checkpoint_Path);
        if latest_Checkpoint is None:
            print('There is no checkpoint.');
            return;

        self.tf_Saver.restore(self.tf_Session, latest_Checkpoint);
        print('Checkpoint \'{}\' is loaded.'.format(latest_Checkpoint));

    def Train(self):
        def Run_Inference():
            with open('Speaker_Embedding_Inference_in_Train.txt', 'r') as f:
                label_List, path_List = list(zip(*[line.strip().split('\t') for line in f.readlines()]))
            self.Inference(path_List= path_List, export_TSNE= True, label_List= label_List)

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
                'Learning rate: {:0.6f}'.format(result_Dict['Learning_Rate']),
                'Loss: {:0.5f}'.format(result_Dict['Loss']),
                ]
            print('\t\t'.join(display_List))

            #global step의 update 타이밍이 좀 바뀐거 같기도 하고??????
            if (result_Dict['Global_Step'] + 1) % hp.Speaker_Embedding.Train.Checkpoint_Save_Timing == 0:
                os.makedirs(os.path.join(hp.Speaker_Embedding.Checkpoint_Path).replace("\\", "/"), exist_ok= True);
                self.tf_Saver.save(self.tf_Session, os.path.join(hp.Speaker_Embedding.Checkpoint_Path, 'SE_CHECKPOINT').replace('\\', '/'), global_step= result_Dict['Global_Step'] + 1);
            if (result_Dict['Global_Step'] + 1) % hp.Speaker_Embedding.Train.Inference_Timing == 0:
                Run_Inference();

    def Inference(self, path_List, export_TSNE= False, label_List= None):
        result_Dict_List = []
        for feed_Dict in self.feeder.Get_Inference_Pattern(path_List):
            result_Dict_List.append(self.tf_Session.run(
                fetches= self.inference_Tensor_Dict,
                feed_dict= feed_Dict
                ))

        result_Embedding = np.vstack([x['Embedding'] for x in result_Dict_List])
        
        if export_TSNE:
            export_TSNE_Thread = Thread(
                target= self.Export_TSNE,
                args= [
                    result_Embedding,
                    label_List,
                    result_Dict_List[0]['Global_Step'],
                    'GS_{}'.format(result_Dict_List[0]['Global_Step'])
                    ]
                )
            export_TSNE_Thread.daemon = True;
            export_TSNE_Thread.start();

        return result_Embedding

    def Export_TSNE(self, embedding_Array, label_List, global_Step, file_Name='TSNE'):
        os.makedirs(os.path.join(hp.Speaker_Embedding.Train.Inference_Path, 'PLOT').replace("\\", "/"), exist_ok= True);

        #Embedding value for comparison
        label_Set = set(label_List)
        mean_Embedding_Dict = {label: np.mean([y for x, y in zip(label_List, embedding_Array) if x == label], axis=0) for label in label_Set}

        between_List = []
        for label1, mean_Embedding1 in mean_Embedding_Dict.items():
            for label2, mean_Embedding2 in mean_Embedding_Dict.items():
                if label1 == label2:
                    continue;
                between_List.append(np.sqrt(np.sum(np.power(mean_Embedding1 - mean_Embedding2, 2))))

        within_List = []
        for label, embedding in zip(label_List, embedding_Array):
            within_List.append(np.sqrt(np.sum(np.power(embedding - mean_Embedding_Dict[label], 2))))

        embedding_Value = np.mean(between_List) / np.mean(within_List)

        #Embedding t-SNE
        #https://www.scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
        display_Speaker_List = list(set(label_List))[:10];  #Max speaker = 10
        color_List = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'];

        tsne = TSNE(n_components=2, random_state=0);
        embedding_Result_2D = tsne.fit_transform(embedding_Array); #[Pattern, 2]        
        fig = plt.figure(figsize=(10, 10));
        for speaker, color in zip(display_Speaker_List, color_List):
            plt.scatter(
                x= embedding_Result_2D[[speaker == x for x in label_List], 0],
                y= embedding_Result_2D[[speaker == x for x in label_List], 1],
                c= color,
                edgecolors = 'k',
                label= speaker)
        plt.title('Global step: {}    t-SNE Result    V: {:0.5f}'.format(global_Step, embedding_Value))
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            os.path.join(hp.Speaker_Embedding.Train.Inference_Path, 'PLOT', '{}.PNG'.format(file_Name)).replace("\\", "/"),
            )
        plt.close(fig)

if __name__ == '__main__':
    new_Speaker_Embedding = Speaker_Embedding(is_Training= True)
    new_Speaker_Embedding.Restore()
    new_Speaker_Embedding.Train()