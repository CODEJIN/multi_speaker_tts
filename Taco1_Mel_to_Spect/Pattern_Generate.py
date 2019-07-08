import numpy as np;
import re, os, librosa, argparse;
from Audio import *;
import _pickle as pickle;
from concurrent.futures import ThreadPoolExecutor as PE;
import Hyper_Parameters as hp;
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']];

def Pattern_File_Generate(path, display_Prefix = ''):
    sig = librosa.core.load(
        path,
        sr = hp.Sound.Sample_Rate
        )[0]
    sig = librosa.effects.trim(sig, top_db= 15, frame_length=32, hop_length=16)[0] * 0.99

    mel = np.transpose(melspectrogram(
        y= sig,
        num_freq= hp.Sound.Spectrogram_Dim,
        frame_shift_ms= hp.Sound.Frame_Shift,
        frame_length_ms= hp.Sound.Frame_Length,
        num_mels= hp.Sound.Mel_Dim,
        sample_rate= hp.Sound.Sample_Rate,
        max_abs_value= hp.Sound.Max_Abs_Mel
        ).astype(np.float32))

    if hp.Taco1_Mel_to_Spect.Train.Max_Mel_Length < mel.shape[0]:
        print('File \'{}\' has too long. This file is ignored.'.format(path))
        return;

    spec = np.transpose(spectrogram(
        y= sig,
        num_freq= hp.Sound.Spectrogram_Dim,
        frame_shift_ms= hp.Sound.Frame_Shift,
        frame_length_ms= hp.Sound.Frame_Length,
        sample_rate= hp.Sound.Sample_Rate
        ).astype(np.float32))

    new_Pattern_Dict = {
        'Spectrogram': spec,
        'Mel': mel,
        }

    pattern_Name = '{}.PICKLE'.format(os.path.splitext(os.path.basename(path))[0]).upper()

    with open(os.path.join(hp.Taco1_Mel_to_Spect.Train.Pattern_Path, pattern_Name).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=2)
            
    print('[{}]'.format(display_Prefix), '{}'.format(path), '->', '{}'.format(pattern_Name).upper())

#VCTK
def VCTK_Info_Load(vctk_Path='E:/Multi_Speaker_TTS.Raw_Data/VCTK'):
    vctk_File_Path_List = [];
    for root, directory_List, file_Name_List in os.walk(vctk_Path):        
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/');
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;
            vctk_File_Path_List.append(wav_File_Path)

#LS
def LS_Info_Load(ls_Path = 'E:/Multi_Speaker_TTS.Raw_Data/LibriSpeech/train'):
    ls_File_Path_List = [];
    for root, directory_List, file_Name_List in os.walk(ls_Path):
        for index, file_Name in enumerate(file_Name_List):
            wav_File_Path = os.path.join(root, file_Name).replace("\\", "/");
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;
            ls_File_Path_List.append(wav_File_Path)

def Metadata_Generate():
    new_Metadata_Dict = {
        'Spectrogram_Dim': hp.Sound.Spectrogram_Dim,
        'Mel_Dim': hp.Sound.Mel_Dim,
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Frame_Length': hp.Sound.Frame_Length,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'File_List': [],
        'Mel_Length_Dict': {},
        }

    for root, dirs, files in os.walk(hp.Taco1_Mel_to_Spect.Train.Pattern_Path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f);
                try:
                    new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                    new_Metadata_Dict['File_List'].append(file)
                except:
                    print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

    with open(os.path.join(hp.Taco1_Mel_to_Spect.Train.Pattern_Path, hp.Taco1_Mel_to_Spect.Train.Metadata_File.upper()).replace('\\', '/'), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=2);

if __name__ == '__main__':
    argParser = argparse.ArgumentParser();
    argParser.add_argument("-vctk", "--vctk_path", required=False);
    argParser.add_argument("-ls", "--ls_path", required=False);

    total_Pattern_Count = 0

    if not argument_Dict['vctk_path'] is None:
        vctk_File_Path_List = VCTK_Info_Load(
            vctk_Path= argument_Dict['vctk_path']
            )
        total_Pattern_Count += len(vctk_File_Path_List)

    if not argument_Dict['ls_path'] is None:
        ls_File_Path_List = LS_Info_Load(
            ls_Path= argument_Dict['ls_path']
            )
        total_Pattern_Count += len(ls_File_Path_List)

    if total_Pattern_Count == 0:
        raise ValueError('Total pattern count is zero.')

    os.makedirs(hp.Taco1_Mel_to_Spect.Train.Pattern_Path, exist_ok= True);
    max_Worker = 10;
    with PE(max_workers = max_Worker) as pe:
        if not argument_Dict['vctk_path'] is None:
            for index, wav_File_Path in enumerate(vctk_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    wav_File_Path,
                    'VCTK {}/{}'.format(index, len(vctk_File_Path_List))
                    )

        if not argument_Dict['ls_path'] is None:
            for index, wav_File_Path in enumerate(ls_File_Path_List):            
                pe.submit(
                    Pattern_File_Generate,
                    wav_File_Path,
                    'LS {}/{}'.format(index, len(ls_File_Path_List))
                    )