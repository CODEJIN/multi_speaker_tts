import tensorflow as tf;
import numpy as np;
import _pickle as pickle;
from collections import deque, Sequence;
from random import shuffle;
from threading import Thread;
import os, time, librosa;
import Hyper_Parameters as hp;
from Audio import melspectrogram, spectrogram

class Feeder:
    def __init__(
        self,
        is_Training= False
        ):
        self.Placeholder_Generate();
        self.Metadata_Load();

        self.pattern_Queue = deque();
        
        pattern_Generate_Thread = Thread(target=self.Train_Pattern_Generate);
        pattern_Generate_Thread.daemon = True;
        pattern_Generate_Thread.start();

    def Placeholder_Generate(self):
        self.placeholder_Dict = {};
        with tf.variable_scope('placeholders') as scope:
            self.placeholder_Dict["Is_Training"] = tf.placeholder(tf.bool, name="is_training_placeholder");    #boolean            
            self.placeholder_Dict["Mel"] = tf.placeholder(tf.float32, shape=(None, None, hp.Sound.Mel_Dim), name="mel_placeholder");    #Shape: [batch_Size, spectrogram_Length, mel_Spectogram_Dimension];
            self.placeholder_Dict["Spectrogram"] = tf.placeholder(tf.float32, shape=(None, None, hp.Sound.Spectrogram_Dim), name="mel_placeholder");    #Shape: [batch_Size, spectrogram_Length, mel_Spectogram_Dimension];
        
    def Metadata_Load(self):
        with open(os.path.join(hp.Taco1_Mel_to_Spect.Train.Pattern_Path, hp.Taco1_Mel_to_Spect.Train.Metadata_File.upper()).replace("\\", "/"), 'rb') as f:
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
        if hp.Taco1_Mel_to_Spect.Train.Pattern_Sorting_by_Length:
            path_List = [
                (path, self.metadata_Dict['Mel_Length_Dict'][path])
                for path in self.metadata_Dict['File_List']
                ]
            path_List = [path for path, _ in sorted(path_List, key=lambda x: x[1])]
        else:
            path_List = self.metadata_Dict['File_List']

        while True:
            if not hp.Taco1_Mel_to_Spect.Train.Pattern_Sorting_by_Length:
                shuffle(path_List)

            path_Batch_List = [
                path_List[x:x + hp.Taco1_Mel_to_Spect.Train.Batch_Size]
                for x in range(0, len(path_List), hp.Taco1_Mel_to_Spect.Train.Batch_Size)
                ]
            shuffle(path_Batch_List)

            batch_Index = 0;
            while batch_Index < len(path_Batch_List):
                if len(self.pattern_Queue) >= hp.Taco1_Mel_to_Spect.Train.Max_Pattern_Queue:
                    time.sleep(0.1);
                    continue;

                pattern_Count = len(path_Batch_List[batch_Index]);

                mel_List = []                
                spectrogram_List = []
                for path in path_Batch_List[batch_Index]:
                    with open(os.path.join(hp.Taco1_Mel_to_Spect.Train.Pattern_Path, path).replace("\\", "/"), "rb") as f:
                        pattern_Dict = pickle.load(f);

                    mel_List.append(pattern_Dict['Mel'])
                    spectrogram_List.append(pattern_Dict['Spectrogram'])

                max_Mel_Length = max([mel.shape[0] for mel in mel_List])
                max_Spectrogram_Length = max([spectrogram.shape[0] for spectrogram in spectrogram_List])

                new_Mel_Pattern = np.zeros(
                    shape=(pattern_Count, max_Mel_Length, hp.Sound.Mel_Dim),
                    dtype= np.float32
                    )
                new_Spectrogram_Pattern = np.zeros(
                    shape=(pattern_Count, max_Spectrogram_Length, hp.Sound.Spectrogram_Dim),
                    dtype= np.float32
                    )

                for pattern_Index, (mel, spect) in enumerate(zip(mel_List, spectrogram_List)):                    
                    new_Mel_Pattern[pattern_Index, :mel.shape[0]] = mel;
                    new_Spectrogram_Pattern[pattern_Index, :spect.shape[0]] = spect;

                self.pattern_Queue.append({
                    self.placeholder_Dict["Is_Training"]: True,
                    self.placeholder_Dict["Mel"]: new_Mel_Pattern,
                    self.placeholder_Dict["Spectrogram"]: new_Spectrogram_Pattern
                    })

                batch_Index += 1;

    def Get_Train_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01);
        return self.pattern_Queue.popleft();

    def Get_Inference_Pattern(self, speaker_Wav_Paths):
        '''
        speaker_Wav_Paths: str or list. If the length is different from 'texts', ValueError will be raised.
        texts: str or list. If the length is different from 'speaker_Wav_Paths', ValueError will be raised.
        if one parameter is str and other is list, str become the list which same value is replicated.
        '''
        if isinstance(speaker_Wav_Paths, str):
            speaker_Wav_Paths = [speaker_Wav_Paths]

        mel_List = [
            np.transpose(melspectrogram(
                y= librosa.effects.trim(librosa.core.load(path, sr = hp.Sound.Sample_Rate)[0], frame_length=32, hop_length=16)[0] * 0.99,
                num_freq= hp.Sound.Spectrogram_Dim,
                frame_shift_ms= hp.Sound.Frame_Shift,
                frame_length_ms= hp.Sound.Frame_Length,
                num_mels= hp.Sound.Mel_Dim,
                sample_rate= hp.Sound.Sample_Rate,
                max_abs_value= hp.Sound.Max_Abs_Mel
                ).astype(np.float32))
            for path in speaker_Wav_Paths
            ]

        mel_Batch_List = [
            mel_List[x:x + hp.Taco1_Mel_to_Spect.Train.Inference.Batch_Size]
            for x in range(0, len(mel_List), hp.Taco1_Mel_to_Spect.Train.Inference.Batch_Size)
            ]
        
        feed_Dict_List= [];
        for mel_Batch in mel_Batch_List:
            pattern_Count = len(mel_Batch)
            max_Mel_Length = max([mel.shape[0] for mel in mel_Batch])

            new_Mel_Pattern = np.zeros(
                shape=(pattern_Count, max_Mel_Length, hp.Sound.Mel_Dim),
                dtype= np.float32
                )

            for mel_Index, mel in enumerate(mel_Batch):
                new_Mel_Pattern[mel_Index, :mel.shape[0]] = mel;

            feed_Dict_List.append({
                self.placeholder_Dict["Is_Training"]: False,
                self.placeholder_Dict["Mel"]: new_Mel_Pattern,
                })

        return feed_Dict_List

if __name__ == '__main__':
    new_Feeder = Feeder()
    x = new_Feeder.Get_Inference_Pattern(speaker_Wav_Paths = [
        'E:/LibriSpeech/test-clean/61/70968/61-70968-0005.flac',            
        'E:/LibriSpeech/test-clean/7021/85628/7021-85628-0027.flac'
        ])
    print(x[0][new_Feeder.placeholder_Dict['Mel']].shape)    
    while True:
        time.sleep(1)
        print(len(new_Feeder.pattern_Queue))