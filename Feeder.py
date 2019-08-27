import tensorflow as tf;
import numpy as np;
import _pickle as pickle;
from collections import deque;
from random import shuffle;
from threading import Thread;
import os, time, librosa, json;
import Hyper_Parameters as hp;
from Audio import melspectrogram

class Feeder:
    def __init__(
        self,
        is_Training= False
        ):
        self.is_Training = is_Training;

        self.Placeholder_Generate();
        self.Metadata_Load();

        if self.is_Training:
            if hp.Train.Use_Pre_in_Main_Train:
                self.pre_Pattern_Queue = deque();
                pre_Pattern_Generate_Thread = Thread(target=self.Train_Pattern_Generate, args=[True]);
                pre_Pattern_Generate_Thread.daemon = True;
                pre_Pattern_Generate_Thread.start();

            self.pattern_Queue = deque();
            pattern_Generate_Thread = Thread(target=self.Train_Pattern_Generate, args=[False]);
            pattern_Generate_Thread.daemon = True;
            pattern_Generate_Thread.start();

    def Placeholder_Generate(self):
        self.placeholder_Dict = {};
        with tf.variable_scope('placeholders') as scope:
            self.placeholder_Dict["Is_Training"] = tf.placeholder(tf.bool, name="is_training_placeholder");    #boolean            
            self.placeholder_Dict["Token"] = tf.placeholder(tf.int32, shape=(None, None, ), name="token_placeholder");    #Shape: [batch_Size, spectrogram_Length, mel_Spectogram_Dimension];
            self.placeholder_Dict["Token_Length"] = tf.placeholder(tf.int32, shape=(None,), name="token_length_placeholder");    #[batch_Size];
            self.placeholder_Dict["Mel"] = tf.placeholder(tf.float32, shape=(None, None, hp.Sound.Mel_Dim), name="mel_placeholder");    #Shape: [batch_Size, spectrogram_Length, mel_Spectogram_Dimension];
            self.placeholder_Dict["Mel_Length"] = tf.placeholder(tf.int32, shape=(None,), name="mel_length_placeholder");    #[batch_Size];
            self.placeholder_Dict['Speaker_Embedding_Mel'] = tf.placeholder(tf.float32, shape=(None, None, hp.Sound.Mel_Dim), name='speaker_embedding_mel_placeholder');    #Shape: [batch_Size, spectrogram_Length, mel_Spectogram_Dimension];
            
    def Metadata_Load(self):
        if self.is_Training:
            with open(os.path.join(hp.Train.Pattern_Path, hp.Train.Metadata_File.upper()).replace("\\", "/"), 'rb') as f:
                self.metadata_Dict = pickle.load(f)

            if not all([
                len(self.metadata_Dict['Token_Index_Dict']) == hp.Encoder.Embedding.Token_Size,
                self.metadata_Dict['Spectrogram_Dim'] == hp.Sound.Spectrogram_Dim,
                self.metadata_Dict['Mel_Dim'] == hp.Sound.Mel_Dim,
                self.metadata_Dict['Frame_Shift'] == hp.Sound.Frame_Shift,
                self.metadata_Dict['Frame_Length'] == hp.Sound.Frame_Length,
                self.metadata_Dict['Sample_Rate'] == hp.Sound.Sample_Rate,
                ]):
                raise ValueError('The metadata information and hyper parameter setting are not consistent.')

        else:
            with open('Token_Index_Dict.json', 'r') as f:
                self.metadata_Dict = {'Token_Index_Dict': json.load(f)}

    def Speaker_Embedding_Mel(self, mel_List):
        required_Mel_Length = \
            hp.Speaker_Embedding.Inference.Sample_Nums * (hp.Speaker_Embedding.Inference.Mel_Frame - hp.Speaker_Embedding.Inference.Overlap_Frame) + \
            hp.Speaker_Embedding.Inference.Overlap_Frame

        new_Mel_Pattern = np.zeros(
            (
                len(mel_List),
                hp.Speaker_Embedding.Inference.Sample_Nums,
                hp.Speaker_Embedding.Inference.Mel_Frame,
                hp.Sound.Mel_Dim
                ),
            dtype=np.float32
            )

        for index, mel in enumerate(mel_List):
            if mel.shape[0] < required_Mel_Length:
                #All sample is same because the mel length is too short.
                sample_Mel = mel[:hp.Speaker_Embedding.Inference.Mel_Frame]
                new_Mel_Pattern[index, :, :sample_Mel.shape[0]] = sample_Mel
            else:
                for sample_Index in range(hp.Speaker_Embedding.Inference.Sample_Nums):
                    start_Point = int((mel.shape[0] - required_Mel_Length) / 2) + sample_Index * hp.Speaker_Embedding.Inference.Overlap_Frame   #Middle of mel
                    new_Mel_Pattern[index, sample_Index] = mel[start_Point:start_Point + hp.Speaker_Embedding.Inference.Mel_Frame]

        return np.reshape(new_Mel_Pattern, (-1, hp.Speaker_Embedding.Inference.Mel_Frame, hp.Sound.Mel_Dim))
    
    def Train_Pattern_Generate(self, is_Pre_Train = False):
        if is_Pre_Train:
            file_List = [path for path in self.metadata_Dict['File_List'] if self.metadata_Dict['Dataset_Dict'][path] in hp.Train.Pre_Train_Dataset_List]
            pattern_Queue = self.pre_Pattern_Queue
        else:
            file_List = [path for path in self.metadata_Dict['File_List'] if self.metadata_Dict['Dataset_Dict'][path] in hp.Train.Main_Train_Dataset_List]
            pattern_Queue = self.pattern_Queue

        min_Mel_Length = hp.Train.Use_Wav_Length_Range[0] / hp.Sound.Frame_Shift
        max_Mel_Length = hp.Train.Use_Wav_Length_Range[1] / hp.Sound.Frame_Shift
        path_List = [
            (path, self.metadata_Dict['Mel_Length_Dict'][path])
            for path in file_List
            if self.metadata_Dict['Mel_Length_Dict'][path] >= min_Mel_Length and self.metadata_Dict['Mel_Length_Dict'][path] <= max_Mel_Length
            ]
        print(
            'Pre train pattern info' if is_Pre_Train else 'Main train pattern info', '\n',
            'Total pattern count: {}'.format(len(self.metadata_Dict['Mel_Length_Dict'])), '\n',
            'Use pattern count: {}'.format(len(path_List)), '\n',
            'Excluded pattern count: {}'.format(len(self.metadata_Dict['Mel_Length_Dict']) - len(path_List))
            )

        if hp.Train.Pattern_Sorting_by_Mel_Length:
            path_List = [file_Name for file_Name, _ in sorted(path_List, key=lambda x: x[1])]
        else:
            path_List = [file_Name for file_Name, _ in path_List]

        while True:
            if not hp.Train.Pattern_Sorting_by_Mel_Length:
                shuffle(path_List)

            path_Batch_List = [
                path_List[x:x + hp.Train.Batch_Size]
                for x in range(0, len(path_List), hp.Train.Batch_Size)
                ]
            shuffle(path_Batch_List)
            #path_Batch_List = path_Batch_List[0:2] + list(reversed(path_Batch_List))  #Batch size의 적절성을 위한 코드. 10회 이상 되면 문제 없음

            batch_Index = 0;
            while batch_Index < len(path_Batch_List):
                if len(pattern_Queue) >= hp.Train.Max_Pattern_Queue:
                    time.sleep(0.1);
                    continue;

                pattern_Count = len(path_Batch_List[batch_Index]);

                token_List = []
                mel_List = []
                for file_Path in path_Batch_List[batch_Index]:
                    with open(os.path.join(hp.Train.Pattern_Path, file_Path).replace("\\", "/"), "rb") as f:
                        pattern_Dict = pickle.load(f);
            
                    token_List.append(np.hstack([
                        self.metadata_Dict['Token_Index_Dict']['<S>'],
                        pattern_Dict['Token'],
                        self.metadata_Dict['Token_Index_Dict']['<E>']
                        ]))
                    mel_List.append(pattern_Dict['Mel'])

                max_Token_Length = max([token.shape[0] for token in token_List])
                max_Mel_Length = max([mel.shape[0] for mel in mel_List])

                new_Token_Pattern = np.zeros(
                    shape=(pattern_Count, max_Token_Length),
                    dtype= np.int32
                    )
                new_Token_Pattern += self.metadata_Dict['Token_Index_Dict']['<E>']  #I think this is useless...
                new_Mel_Pattern = np.zeros(
                    shape=(pattern_Count, max_Mel_Length, hp.Sound.Mel_Dim),
                    dtype= np.float32
                    )

                for pattern_Index, (token, mel) in enumerate(zip(token_List, mel_List)):                    
                    new_Token_Pattern[pattern_Index, :token.shape[0]] = token;
                    new_Mel_Pattern[pattern_Index, :mel.shape[0]] = mel;
                
                pattern_Queue.append({
                    self.placeholder_Dict["Is_Training"]: True,
                    self.placeholder_Dict["Token"]: new_Token_Pattern,
                    self.placeholder_Dict["Token_Length"]: np.array([token.shape[0] for token in token_List]).astype(np.int32),
                    self.placeholder_Dict["Mel"]: new_Mel_Pattern,
                    self.placeholder_Dict["Mel_Length"]: np.array([mel.shape[0] for mel in mel_List]).astype(np.int32),
                    self.placeholder_Dict['Speaker_Embedding_Mel']: self.Speaker_Embedding_Mel(mel_List),
                    })

                batch_Index += 1;

    def Get_Train_Pattern(self, is_Pre_Train = False):
        if is_Pre_Train:
            pattern_Queue = self.pre_Pattern_Queue
        else:
            pattern_Queue = self.pattern_Queue

        while len(pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01);
        return pattern_Queue.popleft();

    def Get_Inference_Pattern(self, speaker_Wav_Path_List, text_List):
        pattern_Count = len(text_List)

        token_List = [
            np.array(
                [self.metadata_Dict['Token_Index_Dict']['<S>']] +
                [self.metadata_Dict['Token_Index_Dict'][letter] for letter in text.upper()] +
                [self.metadata_Dict['Token_Index_Dict']['<E>']]
                ).astype(np.int32)
            for text in text_List
            ]

        max_Token_Length = max([token.shape[0] for token in token_List])

        new_Token_Pattern = np.zeros(
            shape=(pattern_Count, max_Token_Length),
            dtype= np.int32
            )
        new_Token_Pattern += self.metadata_Dict['Token_Index_Dict']['<E>']  #I think this is useless...
        new_Mel_Pattern = np.zeros(
            shape=(pattern_Count, 1, hp.Sound.Mel_Dim),
            dtype= np.float32
            )

        for pattern_Index, token in enumerate(token_List):                    
            new_Token_Pattern[pattern_Index, :token.shape[0]] = token;

        speaker_Embedding_Mel_List = [
            np.transpose(melspectrogram(
                y= librosa.effects.trim(librosa.core.load(path, sr = hp.Sound.Sample_Rate)[0], top_db=15, frame_length=32, hop_length=16)[0] * 0.99,
                num_freq= hp.Sound.Spectrogram_Dim,
                frame_shift_ms= hp.Sound.Frame_Shift,
                frame_length_ms= hp.Sound.Frame_Length,
                num_mels= hp.Sound.Mel_Dim,
                sample_rate= hp.Sound.Sample_Rate,
                max_abs_value= hp.Sound.Max_Abs_Mel
                ).astype(np.float32))
            for path in speaker_Wav_Path_List
            ]

        return {
            self.placeholder_Dict["Is_Training"]: False,
            self.placeholder_Dict["Token"]: new_Token_Pattern,
            self.placeholder_Dict["Token_Length"]: np.array([token.shape[0] for token in token_List]).astype(np.int32),
            self.placeholder_Dict["Mel"]: new_Mel_Pattern,
            self.placeholder_Dict["Mel_Length"]: np.array([0 for _ in text_List]).astype(np.int32),
            self.placeholder_Dict['Speaker_Embedding_Mel']: self.Speaker_Embedding_Mel(speaker_Embedding_Mel_List), 
            }