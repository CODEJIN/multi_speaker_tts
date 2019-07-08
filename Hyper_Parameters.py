import Hyper_Parameters;
import tensorflow as tf;

Sound = tf.contrib.training.HParams(**{
    'Sample_Rate': 16000,
    'Spectrogram_Dim': 1025,
    'Mel_Dim': 80,
    'Max_Abs_Mel': 4,   #If 'None', non symmetric '0 to 1'.
    'Frame_Shift': 12.5,
    'Frame_Length': 50
    })

Encoder = tf.contrib.training.HParams(**{
    'Embedding': tf.contrib.training.HParams(**{
        'Token_Size': 42,
        'Embedding_Size': 512,
        }),
    'Conv': tf.contrib.training.HParams(**{
        'Nums': 3,
        'Kernel_Size': 5,
        'Stride': 1,
        'Channel': 512,
        'Dropout_Rate': 0.5,
        }),
    'BiLSTM': tf.contrib.training.HParams(**{
        'Nums': 1,
        'Cell_Size': 256,   #Uni direction size
        'Zoneout_Rate': 0.1
        }),
    })

Attention = tf.contrib.training.HParams(**{
    'Memory_Size': 128,
    'Conv': tf.contrib.training.HParams(**{
        'Kernel_Size': 31,
        'Stride': 1,
        'Channel': 32,
        'Dropout_Rate': 0.5,
        }),
    })

Decoder = tf.contrib.training.HParams(**{
    'PreNet': tf.contrib.training.HParams(**{
        'Nums': 2,
        'Size': 256,
        'Use_Dropout': True,    #This is independent from 'is_training'
        'Dropout_Rate': 0.5     #if not Use_Dropout, it will be ignored
        }),
    'LSTM': tf.contrib.training.HParams(**{
        'Nums': 2,
        'Cell_Size': 1024,
        'Zoneout_Rate': 0.1,
        'Max_Inference_Length': 1000  #To force stop in inference
        }),
    'Conv': tf.contrib.training.HParams(**{
        'Nums': 5,
        'Kernel_Size': 5,
        'Stride': 1,
        'Channel': 512,
        'Dropout_Rate': 0.5,
        }),    
    })

Train = tf.contrib.training.HParams(**{
    'Pre_Step': 0,
    'Use_Pre_in_Main_Train': False,
    'Pattern_Path': 'E:/MSTTS_SV.Data',
    'Metadata_File': 'METADATA.PICKLE',
    'Batch_Size': 32,
    'Pattern_Sorting_by_Mel_Length': True,
    'Use_Wav_Length_Range': (500, 9000), #(ms), 1000ms = 1sec
    'Pre_Train_Dataset_List': ['LJ'],
    'Main_Train_Dataset_List': ['VCTK', 'TIMIT'],
    'Max_Pattern_Queue': 20,
    'Learning_Rate': tf.contrib.training.HParams(**{
        'Initial': 1e-3,
        'Min': 1e-5,
        'Decay_Start_Step': 0,#9000,
        'Decay_Step': 10000,
        'Decay_Rate': 0.5,
        }),
    'Weight_Regularization_Rate': 1e-6,
    'ADAM': tf.contrib.training.HParams(**{
        'Beta1': 0.9,
        'Beta2': 0.999,
        'Epsilon': 1e-6,
        }),
    'Use_L1_Loss': False,
    'Inference_Timing': 1000,
    'Checkpoint_Save_Timing': 1000
    })

Speaker_Embedding = tf.contrib.training.HParams(**{
    'Embedding_Size': 256,
    'LSTM': tf.contrib.training.HParams(**{
        'Nums': 3,
        'Cell_Size': 256, #768,
        'Zoneout_Rate': 0.1,
        'Use_Residual': True,
        }),
    'Inference': tf.contrib.training.HParams(**{
        'Sample_Nums': 5,
        'Mel_Frame': 64,
        'Overlap_Frame': 32,
        'Max_Embedding_per_Batch': 128,
        }),
    'Checkpoint_Path': 'E:/Speaker_Embedding/Checkpoint',
    'Train': tf.contrib.training.HParams(**{
        'Pattern_Path': 'E:/Speaker_Embedding.Data',
        'Metadata_File': 'METADATA.PICKLE',
        'Batch_Speaker': 32,
        'Batch_per_Speaker': 10,
        'Max_Pattern_Queue': 20,
        'Frame_Range': (140, 180),
        'Loss_Calc_Method': 'Softmax', #'Contrast'
        'Learning_Rate': tf.contrib.training.HParams(**{
            'Initial': 1e-3,
            'Min': 1e-5,
            'Decay_Step': 10000,
            'Decay_Rate': 0.5,
            }),
        'ADAM': tf.contrib.training.HParams(**{
            'Beta1': 0.9,
            'Beta2': 0.999,
            'Epsilon': 1e-8,
            }),
        'Inference_Path': 'E:/Speaker_Embedding',
        'Inference_Timing': 1000,
        'Checkpoint_Save_Timing': 1000
        })
    })

Taco1_Mel_to_Spect = tf.contrib.training.HParams(**{
    'ConvBank': tf.contrib.training.HParams(**{
        'Nums': 1,
        'Max_Kernel_Size': 8,
        'Stride': 1,
        'Channel': 128,
        'Pooling': tf.contrib.training.HParams(**{
            'Size': 2,
            'Stride': 1,
            }),
        'Projection1': tf.contrib.training.HParams(**{
            'Kernel_Size': 3,
            'Stride': 1,
            'Channel': 256,
            }),
        'Projection2': tf.contrib.training.HParams(**{
            'Kernel_Size': 3,
            'Stride': 1,
            'Channel': 80,
            }),
        }),
    'Highway': tf.contrib.training.HParams(**{
        'Nums': 4
        }),
    'BiRNN': tf.contrib.training.HParams(**{
        'Nums': 1,
        'Cell_Size': 128,   #Uni direction size
        'Zoneout_Rate': 0.1
        }),
    'Griffin_Lim_Iteration': 100,
    'Checkpoint_Path': 'E:/Taco1_Mel_to_Spect/Checkpoint',
    })

Inference_Path = 'E:/MSTTS_SV.NoPre.VCTK.TIMIT'
Checkpoint_Path = 'E:/MSTTS_SV.NoPre.VCTK.TIMIT/Checkpoint'