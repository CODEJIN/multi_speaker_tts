import librosa, re, os, json, tempfile, argparse
import numpy as np
import Hyper_Parameters as hp
import _pickle as pickle
from concurrent.futures import ThreadPoolExecutor as PE;
from sphfile import SPHFile
from random import shuffle
from Audio import spectrogram, melspectrogram

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
regex_Checker =  re.compile('[A-Z\'\",.?!\-&;:()\[\]\s]+');
max_Worker= 10

def Text_Filtering(text):
    remove_Letter_List = ['"', ')',]
    replace_List = [(' ?', '?'), ('  ', ' '), (' ,', ','), (' !', '!'),]

    text = text.upper().strip();
    for filter in remove_Letter_List:
        text= text.replace(filter, '')
    for filter, replace_STR in replace_List:
        text= text.replace(filter, replace_STR)

    text= text.strip()

    if len(regex_Checker.findall(text)) > 1:
        return None
    elif text.startswith('\''):
        return None
    else:
        return regex_Checker.findall(text)[0]

def Mel_Generate(path, spectral_Subtract = False, range_Ignore = False):
    sig = librosa.core.load(
        path,
        sr = hp.Sound.Sample_Rate
        )[0]
    sig = librosa.effects.trim(sig, top_db= 15)[0] * 0.99

    sig_Length = sig.shape[0] / hp.Sound.Sample_Rate * 1000  #ms
    if not range_Ignore and (sig_Length < hp.Train.Use_Wav_Length_Range[0] or sig_Length > hp.Train.Use_Wav_Length_Range[1]):
        return None;

    mel = np.transpose(melspectrogram(
        y= sig,
        num_freq= hp.Sound.Spectrogram_Dim,
        frame_shift_ms= hp.Sound.Frame_Shift,
        frame_length_ms= hp.Sound.Frame_Length,
        num_mels= hp.Sound.Mel_Dim,
        sample_rate= hp.Sound.Sample_Rate,
        max_abs_value= hp.Sound.Max_Abs_Mel,
        spectral_subtract= spectral_Subtract
        ).astype(np.float32))

    return mel

def Pattern_File_Generate(path, text, token_Index_Dict, dataset, spectral_Subtract = False, file_Prefix='', display_Prefix = '', range_Ignore = False):
    mel = Mel_Generate(path, spectral_Subtract, range_Ignore)

    if mel is None:
        print('[{}]'.format(display_Prefix), '{}'.format(path), '->', 'Ignored because of length.')
        return

    token = np.array([token_Index_Dict[letter] for letter in text]).astype(np.int32)
    
    new_Pattern_Dict = {
        'Token': token,
        'Mel': mel,
        'Text': text,
        'Dataset': dataset,
        }

    pickle_File_Name = '{}.{}{}.PICKLE'.format(dataset, file_Prefix, os.path.splitext(os.path.basename(path))[0]).upper()

    with open(os.path.join(hp.Train.Pattern_Path, pickle_File_Name).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=2)
            
    print('[{}]'.format(display_Prefix), '{}'.format(path), '->', '{}'.format(pickle_File_Name))

def Pattern_File_Generate_from_SPH(path, text_List, token_Index_Dict, dataset, spectral_Subtract = False, display_Prefix = '', range_Ignore = False):
    sph_Loader = SPHFile(path)
    
    os.makedirs(os.path.join(tempfile.gettempdir(), 'mstts').replace('\\', '/'), exist_ok= True)
    for index, (start_Time, end_Time, text) in enumerate(text_List):
        temp_Wav_Path = os.path.join(
            tempfile.gettempdir(),
            'mstts',
            '{}.{}.wav'.format(os.path.splitext(os.path.basename(path))[0], index)
            ).replace('\\', '/')
        sph_Loader.write_wav(temp_Wav_Path, start_Time, end_Time)

        mel = Mel_Generate(temp_Wav_Path)
        if mel is None:
            print('[{}]'.format(display_Prefix), '{}    {}-{}'.format(path, start_Time, end_Time), '->', 'Ignored because of length.')
            return

        token = np.array([token_Index_Dict[letter] for letter in text]).astype(np.int32)
    
        new_Pattern_Dict = {
            'Token': token,
            'Mel': mel,
            'Text': text,
            'Dataset': dataset,
            }

        pickle_File_Name = '{}.{}.{}.PICKLE'.format(dataset, os.path.splitext(os.path.basename(path))[0], index).upper()

        with open(os.path.join(hp.Train.Pattern_Path, pickle_File_Name).replace("\\", "/"), 'wb') as f:
            pickle.dump(new_Pattern_Dict, f, protocol=2)
        
        os.remove(temp_Wav_Path)

        print('[{}]'.format(display_Prefix), '{}    {}-{}'.format(path, start_Time, end_Time), '->', '{}'.format(pickle_File_Name))

def LJ_Info_Load(ls_Path='E:/Multi_Speaker_TTS.Raw_Data/LJSpeech'):
    text_Dict = {};
    with open(os.path.join(ls_Path, 'metadata.csv').replace('\\', '/'), 'r', encoding='utf-8-sig') as f:
        for readline in f.readlines():
            raw_Data = [x.strip() for x in readline.split('|')];
            file_Name, text = raw_Data[0], raw_Data[2]
                
            text_Dict[file_Name] = text
            
    lj_File_Path_List = [];
    lj_Text_Dict = {};
    for file_Name, text in text_Dict.items():
        wav_File_Path = os.path.join(ls_Path, 'wavs', '{}.wav'.format(file_Name)).replace('\\', '/');
        text = Text_Filtering(text)

        if not os.path.exists(wav_File_Path):
            continue;
        if text is None:
            continue;

        lj_File_Path_List.append(wav_File_Path)
        lj_Text_Dict[wav_File_Path] = text

    print('LJ info generated.')
    return lj_File_Path_List, lj_Text_Dict

def VCTK_Info_Load(vctk_Path='E:/Multi_Speaker_TTS.Raw_Data/VCTK'):
    vctk_Wav_Path = os.path.join(vctk_Path, 'wav48').replace('\\', '/')
    vctk_Txt_Path = os.path.join(vctk_Path, 'txt').replace('\\', '/')

    vctk_File_Path_List = [];
    vctk_Text_Dict = {};
    for root, directory_List, file_Name_List in os.walk(vctk_Wav_Path):
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/');
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;
            txt_File_Path = wav_File_Path.replace(vctk_Wav_Path, vctk_Txt_Path).replace('wav', 'txt');
            if not os.path.exists(txt_File_Path):
                continue;
            with open(txt_File_Path, 'r') as f:
                text = Text_Filtering(f.read().strip());
            if text is None:
                continue;
            vctk_File_Path_List.append(wav_File_Path)
            vctk_Text_Dict[wav_File_Path] = text

    print('VCTK info generated.')
    return vctk_File_Path_List, vctk_Text_Dict;

def LS_Info_Load(ls_Path = 'E:/Multi_Speaker_TTS.Raw_Data/LibriSpeech/train'):
    ls_File_Path_List = [];
    ls_Text_Dict = {};
    for root, directory_List, file_Name_List in os.walk(ls_Path):
        speaker, text_ID = root.replace('\\', '/').split('/')[-2:]

        txt_File_Path = os.path.join(ls_Path, speaker, text_ID, '{}-{}.trans.txt'.format(speaker, text_ID)).replace('\\', '/');
        if not os.path.exists(txt_File_Path):
            continue;

        with open(txt_File_Path, 'r') as f:
            text_Data = f.readlines();

        text_Dict = {}
        for text_Line in text_Data:
            text_Line = text_Line.strip().split(' ');
            text_Dict[text_Line[0]] = ' '.join(text_Line[1:]);

        for index, file_Name in enumerate(file_Name_List):
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/');
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;
            text = Text_Filtering(text_Dict[os.path.splitext(os.path.basename(wav_File_Path))[0]]);
            if text is None:
                continue;
            ls_File_Path_List.append(wav_File_Path)
            ls_Text_Dict[wav_File_Path] = text

    print('LS info generated.')
    return ls_File_Path_List, ls_Text_Dict;

def TL_Info_Load(tl_Path = 'E:/Multi_Speaker_TTS.Raw_Data/Tedlium'):
    tl_SPH_Path = os.path.join(tl_Path, 'sph').replace('\\', '/')
    tl_STM_Path = os.path.join(tl_Path, 'stm').replace('\\', '/')

    tl_File_Path_List = [];
    tl_Text_List_Dict = {};
    for root, directory_List, file_Name_List in os.walk(tl_SPH_Path):
        for file_Name in file_Name_List:
            sph_File_Path = os.path.join(root, file_Name).replace('\\', '/');
            tl_File_Path_List.append(sph_File_Path)

            stm_File_Path = sph_File_Path.replace(tl_SPH_Path, tl_STM_Path).replace('sph', 'stm');
            if not os.path.exists(stm_File_Path):
                continue;
            with open(stm_File_Path, 'r', encoding='utf-8-sig') as f:
                tl_Text_List_Dict[sph_File_Path] = []
                for line in [x.strip().upper().split(' ') for x in f.readlines()]:
                    if '<UNK>' in line:
                        continue;
                    start_Time, end_Time = float(line[3]), float(line[4])

                    text = Text_Filtering(' '.join(line[6:]).replace(' \'', '\''))
                    if not text is None:
                        tl_Text_List_Dict[sph_File_Path].append((start_Time, end_Time, text))

    print('TL info generated.')
    return tl_File_Path_List, tl_Text_List_Dict

def TIMIT_Info_Load(timit_Path = 'E:/Multi_Speaker_TTS.Raw_Data/TIMIT/TRAIN'):
    timit_File_Path_List = [];
    timit_Text_List_Dict = {};
    for root, directory_List, file_Name_List in os.walk(timit_Path):
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/');
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue;
            txt_File_Path = wav_File_Path.replace('WAV', 'TXT')
            if not os.path.exists(txt_File_Path):
                continue;
            with open(txt_File_Path, 'r') as f:
                text = Text_Filtering(' '.join(f.read().strip().split(' ')[2:]).strip());
            if text is None:
                continue;
            timit_File_Path_List.append(wav_File_Path)
            timit_Text_List_Dict[wav_File_Path] = text

    print('TIMIT info generated.')
    return timit_File_Path_List, timit_Text_List_Dict;

def Metadata_Generate(token_Index_Dict):
    new_Metadata_Dict = {
        'Token_Index_Dict': token_Index_Dict,        
        'Spectrogram_Dim': hp.Sound.Spectrogram_Dim,
        'Mel_Dim': hp.Sound.Mel_Dim,
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Frame_Length': hp.Sound.Frame_Length,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'File_List': [],
        'Token_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Dataset_Dict': {},
        }

    for root, dirs, files in os.walk(hp.Train.Pattern_Path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f);
                try:
                    new_Metadata_Dict['Token_Length_Dict'][file] = pattern_Dict['Token'].shape[0]
                    new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                    new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                    new_Metadata_Dict['File_List'].append(file)
                except:
                    print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

    with open(os.path.join(hp.Train.Pattern_Path, hp.Train.Metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=2)

    print('Metadata generate done.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser();
    argParser.add_argument("-lj", "--lj_path", required=False);
    argParser.add_argument("-vctk", "--vctk_path", required=False);
    argParser.add_argument("-ls", "--ls_path", required=False);
    argParser.add_argument("-tl", "--tl_path", required=False);
    argParser.add_argument("-timit", "--timit_path", required=False);
    argParser.add_argument("-all", "--all_save", action='store_true'); #When this parameter is False, only correct time range patterns are generated.
    argParser.set_defaults(all_save = False);
    argument_Dict = vars(argParser.parse_args());

    with open('Token_Index_Dict.json', 'r') as f:
        token_Index_Dict = json.load(f)
    
    total_Pattern_Count = 0

    if not argument_Dict['lj_path'] is None:
        lj_File_Path_List, lj_Text_Dict = LJ_Info_Load(ls_Path= argument_Dict['lj_path']);
        total_Pattern_Count += len(lj_File_Path_List)
    if not argument_Dict['vctk_path'] is None:
        vctk_File_Path_List, vctk_Text_Dict = VCTK_Info_Load(vctk_Path= argument_Dict['vctk_path'])
        total_Pattern_Count += len(vctk_File_Path_List)
    if not argument_Dict['ls_path'] is None:
        ls_File_Path_List, ls_Text_Dict = LS_Info_Load(ls_Path= argument_Dict['ls_path'])
        total_Pattern_Count += len(ls_File_Path_List)
    if not argument_Dict['tl_path'] is None:
        tl_File_Path_List, tl_Text_List_Dict = TL_Info_Load(tl_Path= argument_Dict['tl_path'])
        total_Pattern_Count += len(tl_File_Path_List)
    if not argument_Dict['timit_path'] is None:
        timit_File_Path_List, timit_Text_List_Dict = TIMIT_Info_Load(timit_Path= argument_Dict['timit_path'])
        total_Pattern_Count += len(timit_File_Path_List)

    if total_Pattern_Count == 0:
        raise ValueError('Total pattern count is zero.')
    
    os.makedirs(hp.Train.Pattern_Path, exist_ok= True);
    total_Generated_Pattern_Count = 0
    with PE(max_workers = max_Worker) as pe:
        if not argument_Dict['lj_path'] is None:
            for index, file_Path in enumerate(lj_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    lj_Text_Dict[file_Path],
                    token_Index_Dict,
                    'LJ',
                    False,
                    '',
                    'LJ {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(lj_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['vctk_path'] is None:
            for index, file_Path in enumerate(vctk_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    vctk_Text_Dict[file_Path],
                    token_Index_Dict,
                    'VCTK',
                    False,
                    '',
                    'VCTK {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(vctk_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['ls_path'] is None:
            for index, file_Path in enumerate(ls_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    ls_Text_Dict[file_Path],
                    token_Index_Dict,
                    'LS',
                    True,
                    '',
                    'LS {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(ls_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['tl_path'] is None:
            for index, file_Path in enumerate(tl_File_Path_List):
                pe.submit(
                    Pattern_File_Generate_from_SPH,
                    file_Path,
                    tl_Text_List_Dict[file_Path],
                    token_Index_Dict,
                    'TL',
                    True,
                    'TL {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(tl_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['timit_path'] is None:
            for index, file_Path in enumerate(timit_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    timit_Text_List_Dict[file_Path],
                    token_Index_Dict,
                    'TIMIT',
                    False,
                    '{}.'.format(file_Path.split('/')[-2]),
                    'TIMIT {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(timit_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

    Metadata_Generate(token_Index_Dict)