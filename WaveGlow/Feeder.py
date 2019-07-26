import tensorflow as tf;
import numpy as np;
import _pickle as pickle;
from collections import deque;
from random import shuffle;
from threading import Thread;
import os, time, librosa;
import Hyper_Parameters as hp;
from Audio import melspectrogram;

class Feeder:
    def __init__(
        self,
        is_Training= False
        ):
        self.Placeholder_Generate();        
        self.pattern_Queue = deque();
                
        #self.Train_Pattern_Generate()
        pattern_Generate_Thread = Thread(target=self.Train_Pattern_Generate);
        pattern_Generate_Thread.daemon = True;
        pattern_Generate_Thread.start();

    def Placeholder_Generate(self):
        self.placeholder_Dict = {};
        with tf.variable_scope('placeholders') as scope:
            self.placeholder_Dict['Mel'] = tf.placeholder(tf.float32, shape=(None, None, hp.Sound.Mel_Dim), name='mel_placeholder');    #Shape: [batch_Size, spectrogram_Length, mel_Spectogram_Dimension];
            self.placeholder_Dict['Audio'] = tf.placeholder(tf.float32, shape=(None, None), name='audio_placeholder');    #Shape: [batch_Size, audio_Length];

    def Train_Pattern_Generate(self):
        path_List = []

        for root, dirs, files in os.walk(hp.WaveGlow.Train.Pattern_Path):
            for file in files:
                if os.path.splitext(file)[1].upper() != '.wav'.upper():
                    continue
                path = os.path.join(root, file).replace('\\', '/');
                path_List.append(path)

        while True:
            shuffle(path_List);
            path_Batch_List = [
                path_List[x:x + hp.WaveGlow.Train.Batch_Size]
                for x in range(0, len(path_List), hp.WaveGlow.Train.Batch_Size)
                ]
            shuffle(path_Batch_List);            

            batch_Index = 0;
            while batch_Index < len(path_Batch_List):
                if len(self.pattern_Queue) >= hp.WaveGlow.Train.Max_Pattern_Queue:
                    time.sleep(0.1);
                    continue;
                pattern_Count = len(path_Batch_List[batch_Index]);

                sig_List = [];
                mel_List = [];

                for file_Path in path_Batch_List[batch_Index]:
                    sig = librosa.core.load(
                        file_Path,
                        sr = hp.WaveGlow.Export_Sample_Rate
                        )[0]
                    sig = librosa.effects.trim(sig, top_db=15, frame_length=32, hop_length=16)[0]
                    sig = sig / np.max(np.abs(sig)) * 0.99

                    if sig.shape[0] > hp.WaveGlow.Train.Max_Signal_Length:
                        start_Point = np.random.randint(0, sig.shape[0] - hp.WaveGlow.Train.Max_Signal_Length);
                        sig= sig[start_Point:start_Point + hp.WaveGlow.Train.Max_Signal_Length];
                    else:
                        sig = np.concatenate(
                           [sig, np.zeros((hp.WaveGlow.Train.Max_Signal_Length - sig.shape[0]))],
                           axis=-1
                           )

                    sig_List.append(sig);

                for sig in sig_List:
                    if hp.WaveGlow.Export_Sample_Rate != hp.Sound.Sample_Rate:
                        sig = librosa.core.resample(sig, hp.WaveGlow.Export_Sample_Rate, hp.Sound.Sample_Rate)

                    mel = np.transpose(melspectrogram(
                        y= sig,
                        num_freq= hp.Sound.Spectrogram_Dim,
                        frame_shift_ms= hp.Sound.Frame_Shift,
                        frame_length_ms= hp.Sound.Frame_Length,
                        num_mels= hp.Sound.Mel_Dim,
                        sample_rate= hp.Sound.Sample_Rate,
                        max_abs_value= hp.Sound.Max_Abs_Mel
                        ))
                    
                    mel_List.append(mel);

                max_Sig_Length = max([sig.shape[0] for sig in sig_List])
                max_Mel_Length = max([mel.shape[0] for mel in mel_List])

                new_Sig_Pattern =  np.zeros(
                    shape= (pattern_Count, max_Sig_Length),
                    dtype= np.float32
                    )
                new_Mel_Pattern = np.zeros(
                    shape=(pattern_Count, max_Mel_Length, hp.Sound.Mel_Dim),
                    dtype= np.float32
                    )

                for pattern_Index, (sig, mel) in enumerate(zip(sig_List, mel_List)):                                        
                    new_Sig_Pattern[pattern_Index, :sig.shape[0]] = sig;
                    new_Mel_Pattern[pattern_Index, :mel.shape[0]] = mel;
                    
                self.pattern_Queue.append({
                    self.placeholder_Dict['Audio']: new_Sig_Pattern,
                    self.placeholder_Dict['Mel']: new_Mel_Pattern,
                    })

                batch_Index += 1;
               
    def Get_Train_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01);
        return self.pattern_Queue.popleft();

    def Get_Inference_Pattern(self, path_List):
        original_Sig_List = [];
        feed_Dict_List= [];

        mel_List = [];
        mel_Index_List = [];
        for path in path_List:
            sig = librosa.core.load(
                path,
                sr = hp.WaveGlow.Export_Sample_Rate
                )[0]
            sig = librosa.effects.trim(sig, top_db=15, frame_length=32, hop_length=16)[0]
            sig = sig / np.max(np.abs(sig)) * 0.99
            original_Sig_List.append(sig)

            if hp.WaveGlow.Export_Sample_Rate != hp.Sound.Sample_Rate:
                sig = librosa.core.resample(sig, hp.WaveGlow.Export_Sample_Rate, hp.Sound.Sample_Rate)

            mel = np.transpose(melspectrogram(
                y= sig,
                num_freq= hp.Sound.Spectrogram_Dim,
                frame_shift_ms= hp.Sound.Frame_Shift,
                frame_length_ms= hp.Sound.Frame_Length,
                num_mels= hp.Sound.Mel_Dim,
                sample_rate= hp.Sound.Sample_Rate,
                max_abs_value= hp.Sound.Max_Abs_Mel
                ))

            split_Mel_List = [
                mel[x:x+hp.WaveGlow.Inference.Mel_Split_Length]
                for x in range(0, mel.shape[0], hp.WaveGlow.Inference.Mel_Split_Length)
                ]
            mel_List.extend(split_Mel_List)

            start_Index = 0 if len(mel_Index_List) == 0 else mel_Index_List[-1][1]
            mel_Index_List.append((start_Index, start_Index + len(split_Mel_List)))

        pattern_Count = len(mel_List)
        max_Mel_Length = max([mel.shape[0] for mel in mel_List])

        new_Mel_Pattern = np.zeros(
            shape=(pattern_Count, max_Mel_Length, hp.Sound.Mel_Dim),
            dtype= np.float32
            )
        for pattern_Index, mel in enumerate(mel_List):
            new_Mel_Pattern[pattern_Index, :mel.shape[0]] = mel;

        for batch_Start_Index in range(0, pattern_Count, hp.WaveGlow.Inference.Batch_Size):
            feed_Dict_List.append({
                self.placeholder_Dict['Mel']: new_Mel_Pattern[batch_Start_Index:batch_Start_Index + hp.WaveGlow.Inference.Batch_Size],
                })

        return original_Sig_List, feed_Dict_List, mel_Index_List

if __name__ == '__main__':
    with open('WaveGlow_Inference_File_Path_in_Train.txt', 'r') as f:
        path_List = [path.strip() for path in f.readlines()]

    new_Feeder = Feeder()
    x = new_Feeder.Get_Inference_Pattern(path_List)
    for index in range(11):
        print(x[1][index][new_Feeder.placeholder_Dict['Mel']].shape)
    assert False;