import numpy as np;
import re, os, librosa, argparse;
from Audio import *;
import _pickle as pickle;
from concurrent.futures import ProcessPoolExecutor as PE;
import Hyper_Parameters as hp;
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']];

'''
Making pickle the wav file and speaker information.
Each pickle contains two pieces of data: Speaker, Mel
Metadata.pickle contains speaker information, details of the entire pickle, and a pickle list.
'''
max_Worker = 10;

def Pickle_Generate(lexicon_Name, pattern_Index, speaker, file_Path):    
    sig = librosa.core.load(
        file_Path,
        sr = hp.Sound.Sample_Rate
        )[0];

    mel = melspectrogram(
        y= sig,
        num_freq= hp.Sound.Spectrogram_Dim,
        frame_shift_ms= hp.Sound.Frame_Shift,
        frame_length_ms= hp.Sound.Frame_Length,
        num_mels= hp.Sound.Mel_Dim,
        sample_rate= hp.Sound.Sample_Rate,
        max_abs_value= hp.Sound.Max_Abs_Mel
        ).astype(np.float32)

    new_Pattern_Dict = {
        'Speaker': speaker,
        'Mel': np.transpose(mel)
        }    

    pattern_File_Name = '{}.{:07d}.pickle'.format(lexicon_Name, pattern_Index)
    with open(os.path.join(hp.Speaker_Embedding.Train.Pattern_Path, pattern_File_Name).replace('\\', '/'), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=2);
            
    print('{}    {}    ->    {}'.format(lexicon_Name, file_Path, pattern_File_Name));

def Pattern_Generate_VCTK(wav_Path):    
    os.makedirs(hp.Speaker_Embedding.Train.Pattern_Path, exist_ok=True);
        
    print('VCTK raw file list generating...');
    file_List = [];
    for root, directory_List, file_Name_List in os.walk(wav_Path):
        speaker = 'VCTK.{}'.format(root.replace('\\', '/').split('/')[-1]);
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/');
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;            
            file_List.append((speaker, wav_File_Path))
    print('VCTK raw file list generating...Done');

    with PE(max_workers = max_Worker) as pe:
        for pattern_Index, (speaker, wav_File_Path) in enumerate(file_List):
            pe.submit(Pickle_Generate, 'VCTK', pattern_Index, speaker, wav_File_Path);

def Pattern_Generate_LS(data_Path):
    os.makedirs(hp.Speaker_Embedding.Train.Pattern_Path, exist_ok=True);

    print('LS raw file list generating...');
    file_List = [];
    for root, directory_List, file_Name_List in os.walk(data_Path):
        speaker, _ = root.replace('\\', '/').split('/')[-2:];
        speaker = 'LS.{}'.format(speaker);
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/');
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;
            file_List.append((speaker, wav_File_Path))
    print('LS raw file list generating...Done');

    with PE(max_workers = max_Worker) as pe:
        for pattern_Index, (speaker, wav_File_Path) in enumerate(file_List):
            pe.submit(Pickle_Generate, 'LS', pattern_Index, speaker, wav_File_Path);

def Pattern_Generate_VC(lexicon_Suffix, wav_Path):
    os.makedirs(hp.Speaker_Embedding.Train.Pattern_Path, exist_ok=True);
        
    print('VC{} raw file list generating...'.format(lexicon_Suffix));
    file_List = [];
    for root, directory_List, file_Name_List in os.walk(wav_Path):
        speaker = 'VC{}.{}'.format(lexicon_Suffix, root.replace('\\', '/').split('/')[-2]);
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/');
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;
            file_List.append((speaker, wav_File_Path))
    print('VC{} raw file list generating...Done'.format(lexicon_Suffix));
            
    with PE(max_workers = max_Worker) as pe:
        for pattern_Index, (speaker, wav_File_Path) in enumerate(file_List):
            pe.submit(Pickle_Generate, 'VC{}'.format(lexicon_Suffix), pattern_Index, speaker, wav_File_Path);

def Metadata_Generate():
    new_Metadata_Dict = {
        'Sample_Rate': hp.Sound.Sample_Rate,
        'Spectrogram_Dim': hp.Sound.Spectrogram_Dim,
        'Mel_Dim': hp.Sound.Mel_Dim,
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Frame_Length': hp.Sound.Frame_Length,
        'Speaker_File_List_Dict': {},
        'Speaker_Dict': {},
        'Mel_Length_Dict': {},
        }

    #Because ProcessPoolExcuter runs the independent clients, global variables cannot be used and all files must be reloaded for metadata.
    print('Pickle data check...')
    for root, directory_List, file_Name_List in os.walk(hp.Speaker_Embedding.Train.Pattern_Path):
        for index, pattern_File_Name in enumerate(file_Name_List):
            if pattern_File_Name.upper() == hp.Speaker_Embedding.Train.Metadata_File.upper():
                continue;
            with open(os.path.join(root, pattern_File_Name).replace('\\', '/'), 'rb') as f:
                load_Dict = pickle.load(f);
            if not load_Dict['Speaker'] in new_Metadata_Dict['Speaker_File_List_Dict'].keys():
                new_Metadata_Dict['Speaker_File_List_Dict'][load_Dict['Speaker']] = [];
            new_Metadata_Dict['Speaker_File_List_Dict'][load_Dict['Speaker']].append(pattern_File_Name)
            new_Metadata_Dict['Speaker_Dict'][pattern_File_Name] = load_Dict['Speaker'];
            new_Metadata_Dict['Mel_Length_Dict'][pattern_File_Name] = load_Dict['Mel'].shape[0];
            print('{}/{}    {}    Done'.format(index + 1, len(file_Name_List), pattern_File_Name));

    print('Pickle data check...Done')

    with open(os.path.join(hp.Speaker_Embedding.Train.Pattern_Path, hp.Speaker_Embedding.Train.Metadata_File.upper()).replace('\\', '/'), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=2);


if __name__ == '__main__':
    argParser = argparse.ArgumentParser();
    argParser.add_argument("-vctk", "--vctk_path", required=False);
    argParser.add_argument("-ls", "--ls_path", required=False);
    argParser.add_argument("-vox1", "--vox1_path", required=False);
    argParser.add_argument("-vox2", "--vox2_path", required=False);
    argument_Dict = vars(argParser.parse_args());

    if all([argument_Dict[dataset] is None for dataset in ['vctk_path', 'ls_path', 'vox1_path', 'vox2_path']]):
        raise ValueError('At least, the path of one dataset should be assigned.')

    if not argument_Dict['vctk_path'] is None:
        Pattern_Generate_VCTK(
            wav_Path= argument_Dict['vctk_path']
            )
    if not argument_Dict['ls_path'] is None:
        Pattern_Generate_LS(
            data_Path= argument_Dict['ls_path']
            )
    if not argument_Dict['vox1_path'] is None:
        Pattern_Generate_VC(   
            lexicon_Suffix= '1',
            wav_Path= argument_Dict['vox1_path']
            )
    if not argument_Dict['vox2_path'] is None:
        Pattern_Generate_VC(
            lexicon_Suffix= '2',
            wav_Path= argument_Dict['vox2_path']
            )
    Metadata_Generate()
