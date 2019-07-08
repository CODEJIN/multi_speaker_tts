import tensorflow as tf;
import numpy as np;
import _pickle as pickle;
from collections import deque;
from random import shuffle, sample;
from threading import Thread;
from concurrent.futures import ThreadPoolExecutor as PE;
import os, time, librosa;
import Hyper_Parameters as hp;
from Audio import melspectrogram;

class Feeder:
    def __init__(
        self,
        is_Training= False
        ):
        self.Placeholder_Generate(); 
        
        if is_Training:
            self.Metadata_Load();
            self.pattern_Queue = deque();
            pattern_Generate_Thread = Thread(target=self.Train_Pattern_Generate);
            pattern_Generate_Thread.daemon = True;
            pattern_Generate_Thread.start();

    def Placeholder_Generate(self):
        self.placeholder_Dict = {};
        with tf.variable_scope('placeholders') as scope:
            self.placeholder_Dict["Is_Training"] = tf.placeholder(tf.bool, name="is_training_placeholder");    #boolean            
            self.placeholder_Dict["Mel"] = tf.placeholder(tf.float32, shape=(None, None, hp.Sound.Mel_Dim), name="mel_placeholder");    #Shape: [batch_Size, mel_Length, mel_Dimension];
            self.placeholder_Dict["Mel_Length"] = tf.placeholder(tf.uint16, shape=(None,), name="mel_length_placeholder");    #Shape: [batch_Size];

    def Metadata_Load(self):
        with open(os.path.join(hp.Speaker_Embedding.Train.Pattern_Path, hp.Speaker_Embedding.Train.Metadata_File.upper()).replace("\\", "/"), 'rb') as f:
            self.metadata_Dict = pickle.load(f)

        #Consistency check
        if not all([
            self.metadata_Dict['Spectrogram_Dim'] == hp.Sound.Spectrogram_Dim,
            self.metadata_Dict['Mel_Dim'] == hp.Sound.Mel_Dim,
            self.metadata_Dict['Frame_Shift'] == hp.Sound.Frame_Shift,
            self.metadata_Dict['Frame_Length'] == hp.Sound.Frame_Length,
            self.metadata_Dict['Sample_Rate'] == hp.Sound.Sample_Rate,
            ]):
            raise ValueError('The metadata information and hyper parameter setting are not consistent.')

    def Train_Pattern_Generate(self):
        speaker_List = [
            speaker
            for speaker, file_List in self.metadata_Dict['Speaker_File_List_Dict'].items()
            if len(file_List) >= hp.Speaker_Embedding.Train.Batch_per_Speaker
            ]

        while True:
            shuffle(speaker_List);
            speaker_Batch_List = [
                speaker_List[x:x + hp.Speaker_Embedding.Train.Batch_Speaker]
                for x in range(0, len(speaker_List), hp.Speaker_Embedding.Train.Batch_Speaker)
                ]
            shuffle(speaker_Batch_List);            

            batch_Index = 0;
            while batch_Index < len(speaker_Batch_List):
                if len(self.pattern_Queue) >= hp.Speaker_Embedding.Train.Max_Pattern_Queue:
                    time.sleep(0.1);
                    continue;
                speaker_Count = len(speaker_Batch_List[batch_Index]);
                pattern_Count = speaker_Count * hp.Speaker_Embedding.Train.Batch_per_Speaker;

                mel_Length = np.random.randint(
                    low= hp.Speaker_Embedding.Train.Frame_Range[0],
                    high= hp.Speaker_Embedding.Train.Frame_Range[1] + 1
                    )

                new_Mel_Pattern = np.zeros((speaker_Count, hp.Speaker_Embedding.Train.Batch_per_Speaker, mel_Length, hp.Sound.Mel_Dim), dtype=np.float32)

                for speaker_Index, speaker in enumerate(speaker_Batch_List[batch_Index]):
                    sample_Path_List = sample(self.metadata_Dict['Speaker_File_List_Dict'][speaker], hp.Speaker_Embedding.Train.Batch_per_Speaker)
                    for sample_Index, file_Path in enumerate(sample_Path_List):
                        with open(os.path.join(hp.Speaker_Embedding.Train.Pattern_Path, file_Path).replace('\\','/'), 'rb') as f:
                            load_Dict = pickle.load(f)
                            
                        if speaker != load_Dict["Speaker"]:
                            raise ValueError('The speaker labeling of pattern is wrong.\nPattern: {}'.format(file_Path))

                        if load_Dict['Mel'].shape[0] > mel_Length:
                            start_Point = np.random.randint(0, load_Dict['Mel'].shape[0] - mel_Length);
                            new_Mel_Pattern[speaker_Index, sample_Index] = load_Dict['Mel'][start_Point:start_Point + mel_Length];
                        elif load_Dict['Mel'].shape[0] == mel_Length:
                            new_Mel_Pattern[speaker_Index, sample_Index] = load_Dict['Mel'];
                        elif load_Dict['Mel'].shape[0] < mel_Length:
                            start_Point = np.random.randint(0, mel_Length - load_Dict['Mel'].shape[0]);
                            new_Mel_Pattern[speaker_Index, sample_Index, start_Point:start_Point + load_Dict['Mel'].shape[0]] = load_Dict['Mel'];

                self.pattern_Queue.append({
                    self.placeholder_Dict["Is_Training"]: True,
                    self.placeholder_Dict["Mel"]: np.reshape(new_Mel_Pattern, (-1, mel_Length, hp.Sound.Mel_Dim)),
                    self.placeholder_Dict["Mel_Length"]: np.zeros((pattern_Count,)) + mel_Length,
                    })

                batch_Index += 1;
               
    def Get_Train_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.            
            time.sleep(0.01);
        return self.pattern_Queue.popleft();

    def Get_Inference_Pattern(self, path_List):
        required_Mel_Length = \
            hp.Speaker_Embedding.Inference.Sample_Nums * (hp.Speaker_Embedding.Inference.Mel_Frame - hp.Speaker_Embedding.Inference.Overlap_Frame) + \
            hp.Speaker_Embedding.Inference.Overlap_Frame

        feed_Dict_List= [];

        path_Batch_List = [
            path_List[x:x + hp.Speaker_Embedding.Inference.Max_Embedding_per_Batch]
            for x in range(0, len(path_List), hp.Speaker_Embedding.Inference.Max_Embedding_per_Batch)
            ]

        for path_Batch in path_Batch_List:
            new_Mel_Pattern = np.zeros(
                (
                    len(path_Batch),
                    hp.Speaker_Embedding.Inference.Sample_Nums,
                    hp.Speaker_Embedding.Inference.Mel_Frame,
                    hp.Sound.Mel_Dim
                    ),
                dtype=np.float32
                )
            for path_Index, file_Path in enumerate(path_Batch):
                sig = librosa.core.load(
                    file_Path,
                    sr = hp.Sound.Sample_Rate
                    )[0]
                sig = librosa.effects.trim(sig, top_db= 15, frame_length= 32, hop_length= 16)[0] * 0.99
                mel = np.transpose(melspectrogram(
                    y= sig,
                    num_freq= hp.Sound.Spectrogram_Dim,
                    frame_shift_ms= hp.Sound.Frame_Shift,
                    frame_length_ms= hp.Sound.Frame_Length,
                    num_mels= hp.Sound.Mel_Dim,
                    sample_rate= hp.Sound.Sample_Rate,
                    max_abs_value= hp.Sound.Max_Abs_Mel
                    ))

                if mel.shape[0] < required_Mel_Length:
                    #All sample is same because the mel length is too short.
                    sample_Mel = mel[:hp.Speaker_Embedding.Inference.Mel_Frame]
                    new_Mel_Pattern[path_Index, :, :sample_Mel.shape[0]] = sample_Mel
                else:
                    for sample_Index in range(hp.Speaker_Embedding.Inference.Sample_Nums):
                        start_Point = int((mel.shape[0] - required_Mel_Length) / 2) + sample_Index * hp.Speaker_Embedding.Inference.Overlap_Frame
                        new_Mel_Pattern[path_Index, sample_Index] = mel[start_Point:start_Point + hp.Speaker_Embedding.Inference.Mel_Frame]

            feed_Dict_List.append({
                self.placeholder_Dict["Is_Training"]: False,
                self.placeholder_Dict["Mel"]: np.reshape(new_Mel_Pattern, (-1, hp.Speaker_Embedding.Inference.Mel_Frame, hp.Sound.Mel_Dim)),
                self.placeholder_Dict["Mel_Length"]: np.zeros((len(path_Batch) * hp.Speaker_Embedding.Inference.Sample_Nums,)) + hp.Speaker_Embedding.Inference.Mel_Frame,
                })

        return feed_Dict_List
