# Multi speaker TTS

This code is an implementation of the paper 'Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis', __except 'WAVENET'__. The algorithm is based on the following papers:

    Wang, Y., Skerry-Ryan, R. J., Stanton, D., Wu, Y., Weiss, R. J., Jaitly, N., ... & Le, Q. (2017). Tacotron: Towards end-to-end speech synthesis. arXiv preprint arXiv:1703.10135.
    Wan, L., Wang, Q., Papir, A., & Moreno, I. L. (2017). Generalized end-to-end loss for speaker verification. arXiv preprint arXiv:1710.10467.
    Jia, Y., Zhang, Y., Weiss, R. J., Wang, Q., Shen, J., Ren, F., ... & Wu, Y. (2018). Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis. arXiv preprint arXiv:1806.04558.
    
# Structrue
![Structure](https://user-images.githubusercontent.com/17133841/60824607-e49a2c00-a177-11e9-90ea-fb617167871c.png)
The model is divided into three parts that are learned independently of each other: speaker embedding, tacotron 2, and vocoder. Of these, the vocoder is currently using the same structure as Tacotron 1, and may be replaced by Wavenet or Waveglow in the future.

# Used dataset
Currently uploaded code is compatible with the following datasets. The O mark to the left of the dataset name is the dataset actually used in the uploaded result.

## Speaker embedding
    [X] VCTK: https://datashare.is.ed.ac.uk/handle/10283/2651
    [X] LibriSpeech: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
    [O] VoxCeleb: http://www.openslr.org/12/
    
## Mel to Spectrogram
    [O] VCTK: https://datashare.is.ed.ac.uk/handle/10283/2651
    [O] LibriSpeech: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
    
## Multi speaker TTS
    [X] LJSpeech: https://keithito.com/LJ-Speech-Dataset/
    [O] VCTK: https://datashare.is.ed.ac.uk/handle/10283/2651
    [X] LibriSpeech: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
    [X] Tedlium: http://www.openslr.org/12/
    [O] TIMIT: http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3
    
# Instruction
Before proceeding, please set the pattern, inference, and checkpoint paths in 'Hyper_Parameter.py' according to your environment.

## Training

### Speaker embedding

#### Generate pattern
    python -m Speaker_Embedding.Pattern_Generate [options]

    option list:
    -vctk <path>		Set the path of VCTK. VCTK's patterns are generated.
    -ls <path>		Set the path of LibriSpeech. LibriSpeech's patterns are generated.
    -vox1 <path>		Set the path of VoxCeleb1. VoxCeleb1's patterns are generated.
    -vox2 <path>		Set the path of VoxCeleb2. VoxCeleb2's patterns are generated.

#### Set inference files path while training for verification. Edit 'Speaker_Embedding_Inference_in_Train.txt'
    
#### Run
    python -m Speaker_Embedding.Speaker_Embedding

### Mel to spectrogram

#### Generate pattern
    python -m Taco1_Mel_to_Spect.Pattern_Generate [options]

    option list:
    -vctk <path>		Set the path of VCTK. VCTK's patterns are generated.
    -ls <path>		Set the path of LibriSpeech. LibriSpeech's patterns are generated.

#### Set inference files path while training for verification. Edit 'Mel_to_Spect_Inference_in_Train.txt'
    
#### Run
    python -m Taco1_Mel_to_Spect.Taco1_Mel_to_Spect
    
### Multi speaker TTS

#### Generate pattern
    python Pattern_Generate.py [options]

    option list:
    -lj <path>		Set the path of LJSpeech. LJSpeech's patterns are generated.
    -vctk <path>		Set the path of VCTK. VCTK's patterns are generated.
    -ls <path>		Set the path of LibriSpeech. LibriSpeech's patterns are generated.
    -tl <path>		Set the path of Tedlium. Tedlium's patterns are generated.
    -timit <path>		Set the path of TIMIT. TIMIT's patterns are generated.
    -all		All save option. Generator ignore the 'Use_Wav_Length_Range' hyper parameter. If this option is not set, only patterns matching 'Use_Wav_Length_Range' will be generated.

#### Set inference files path and sentence while training for verification. Edit 'Inference_Sentence_in_Train.txt'

#### Run
    python MSTTS_SV.py
    
## Test

### Run 'ipython' in the model's directory.
### Run following command:
    from MSTTS_SV import Tacotron2
    new_Tacotron2 = Tacotron2(is_Training= False)
    new_Tacotron2.Restore()

### Set the speaker's Wav path list and text list like the following example:
    path_List = [
        'E:/Multi_Speaker_TTS.Raw_Data/LJSpeech/wavs/LJ040-0143.wav',
        'E:/Multi_Speaker_TTS.Raw_Data/LibriSpeech/train/17/363/17-363-0039.flac',
        'E:/Multi_Speaker_TTS.Raw_Data/VCTK/wav48/p314/p314_020.wav',
        'E:/Multi_Speaker_TTS.Raw_Data/VCTK/wav48/p256/p256_001.wav'
        ]
    text_List = [
        'He that has no shame has no conscience.',
        'Who knows much believes the less.',
        'Things are always at their best in the beginning.',
        'Please call Stella.'
        ]

__※Two lists should have same length.__

### Run following command:
    new_Tacotron2.Inference(
        path_List = path_List,
        text_List = text_List,
        file_Prefix = 'Result'
        )
    
# Result
## Speaker embedding
![GS_10000](https://user-images.githubusercontent.com/17133841/60827437-30e86a80-a17e-11e9-9d2a-a4620595eaeb.PNG)

## Mel to spectrogram
![GS_12000 IDX_1](https://user-images.githubusercontent.com/17133841/60827448-3ba2ff80-a17e-11e9-97c5-e13f205ebab0.PNG)

## Multi speaker TTS
![GS_131000 IDX_0](https://user-images.githubusercontent.com/17133841/60827468-48275800-a17e-11e9-8fb8-8dc05c3248bf.PNG)

![GS_131000 IDX_1](https://user-images.githubusercontent.com/17133841/60827469-48bfee80-a17e-11e9-920e-a6931dd6cc10.PNG)

![GS_131000 IDX_2](https://user-images.githubusercontent.com/17133841/60827470-48bfee80-a17e-11e9-96c2-868335fcb31c.PNG)

![GS_131000 IDX_3](https://user-images.githubusercontent.com/17133841/60827471-48bfee80-a17e-11e9-8eaa-c9479e2c3e3b.PNG)

![GS_131000 IDX_4](https://user-images.githubusercontent.com/17133841/60827472-48bfee80-a17e-11e9-9759-98769ce8668e.PNG)

[WAV.zip](https://github.com/CODEJIN/multi_speaker_tts/files/3369390/WAV.zip)

# Future works
Training __Waveglow__ for Vocoder Change
