#from keras.models import Model

#from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
#from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

#from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
#from utils.keras_utils import train_model, evaluate_model, set_trainable
#from utils.layer_utils import AttentionLSTM

import numpy as np
from scipy import signal
from tf_classifier import *
import copy
import matplotlib.pyplot as plt
from random import shuffle

classLegs = ts_classifier();
TRAIN_FILES = ["Data/"]


fold_index = 9;
trainData = [];
testData = [];
DataPCA = [];
lisInd = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
shuffle(lisInd);
print(lisInd);
for inde1 in range(0, fold_index):
    inde = lisInd[inde1];
    print("inde train")
    print(inde)
    print(inde1)
    A_train_path_mar = TRAIN_FILES[0] + "A_TXT/%dAmar.txt" % inde;
    lines2 = np.transpose(np.loadtxt(A_train_path_mar, comments="#", delimiter="\t", unpack=False))

    A_train_path_pie = TRAIN_FILES[0] + "A_TXT/%dApie.txt" % inde;
    lines3 = np.transpose(np.loadtxt(A_train_path_pie, comments="#", delimiter="\t", unpack=False))
    A_train_path_sen = TRAIN_FILES[0] + "A_TXT/%dAsen.txt" % inde;
    lines4 = np.transpose(np.loadtxt(A_train_path_sen, comments="#", delimiter="\t", unpack=False))
    lines = [];
    for ind,a in enumerate(lines2):
        secs = round(len(lines2[ind]) / 2000.0);  # Number of seconds in signal X
        samps = secs * 75; # Number of samples to downsample
        #print(samps);
        #samps = 100;
        #print(len(lines2[ind]))
        #print(signal.resample(lines2[ind], samps).shape)

        lines.append(signal.resample(lines2[ind], samps));
        lines.append(signal.resample(lines3[ind], samps));
        lines.append(signal.resample(lines4[ind], samps));

        #lines.append(np.append(signal.resample(lines2[ind], samps),np.array([1])));
        #plt.figure(1)
        #plt.subplot(211)
        #plt.plot(a)
        #plt.subplot(212)
        #plt.plot(lines[-1][:-1])
        #plt.show()
        lines.append(np.append(signal.resample(lines3[ind], samps),np.array([1])));
        #lines.append(np.append(signal.resample(lines4[ind], samps),np.array([1])));
    #for i,lin in enumerate(lines):
    #    if ((lin[-1])!=1):
    #        print("whaaaaatttt")
    #        print(lin[-1])
    #        print(i)
        #else:
        #    lines.append(signal.resample(lines2[ind], samps));
            #lines = np.append(lines,[signal.resample(lines2[ind], samps)],axis=0);
    #print(lines)
        #lines[ind] = block_mean(a, 5).shape  # (20, 40)

    trainData.append(lines);
    #trainData.append(np.transpose(np.load(A_train_path)));

    N_train_path_mar = TRAIN_FILES[0] + "N_TXT/%dNmar.txt" % inde;
    lines2 = np.transpose(np.loadtxt(N_train_path_mar, comments="#", delimiter="\t", unpack=False))
    N_train_path_pie = TRAIN_FILES[0] + "N_TXT/%dNpie.txt" % inde;
    lines3 = np.transpose(np.loadtxt(N_train_path_pie, comments="#", delimiter="\t", unpack=False))
    N_train_path_sen = TRAIN_FILES[0] + "N_TXT/%dNsen.txt" % inde;
    lines4 = np.transpose(np.loadtxt(N_train_path_sen, comments="#", delimiter="\t", unpack=False))
    #lines = copy.deepcopy(lines2);
    lines = [];
    for ind,a in enumerate(lines2):
        secs = round(len(lines2[ind]) / 2000.0);  # Number of seconds in signal X
        samps = secs * 75;  # Number of samples to downsample

        #samps = 100;

        lines.append(signal.resample(lines2[ind], samps));
        lines.append(signal.resample(lines3[ind], samps));
        lines.append(signal.resample(lines4[ind], samps));

        #lines.append(np.append(signal.resample(lines2[ind], samps),np.array([0])));
        #plt.figure(1)
        #plt.subplot(211)
        #plt.plot(a)
        #plt.subplot(212)
        #plt.plot(lines[-1][:-1])
        #plt.show()
        #lines.append(np.append(signal.resample(lines3[ind], samps),np.array([0])));
        #lines.append(np.append(signal.resample(lines4[ind], samps),np.array([0])));
        #if (len(lines)==0):
        #    lines.append(signal.resample(lines2[ind], samps));
        #else:
        #    lines.append(signal.resample(lines2[ind], samps));
        #print("tyep")
        #print(type(lines))
        #lines2[ind] = signal.resample(lines2[ind], samps);
        #print(samps);
    #print(lines2.shape)
        #lines[ind] = block_mean(a, 5).shape  # (20, 40)

    trainData.append(lines)
    DataPCA.append(lines)
    #trainData.append(np.transpose(np.load(N_train_path)));

for inde1 in range(9, 11):
    inde = lisInd[inde1]
    #A_test_path = TRAIN_FILES + "A_TXT/%dAmar.txt" % inde;
    #testData.append(np.transpose(np.load(A_test_path)));

    #N_test_path = TRAIN_FILES + "N_TXT/%dNmar.txt" % inde;
    #testData.append(np.transpose(np.load(N_test_path)));
    print("inde test")
    print(inde)
    print(inde1)
    #print(len(lisInd))
    A_train_path_mar = TRAIN_FILES[0] + "A_TXT/%dAmar.txt" % inde;
    lines2 = np.transpose(np.loadtxt(A_train_path_mar, comments="#", delimiter="\t", unpack=False))
    A_train_path_pie = TRAIN_FILES[0] + "A_TXT/%dApie.txt" % inde;
    lines3 = np.transpose(np.loadtxt(A_train_path_pie, comments="#", delimiter="\t", unpack=False))
    A_train_path_sen = TRAIN_FILES[0] + "A_TXT/%dAsen.txt" % inde;
    lines4 = np.transpose(np.loadtxt(A_train_path_sen, comments="#", delimiter="\t", unpack=False))
    #lines = copy.deepcopy(lines2);
    lines = [];
    for ind,a in enumerate(lines2):
        secs = round(len(lines2[ind]) / 2000.0);  # Number of seconds in signal X
        samps = secs * 75;
        #print(samps);
        #samps = 100;
        #lines2[ind] = signal.resample(lines2[ind],samps);  # Number of samples to downsample


        lines.append(signal.resample(lines2[ind], samps));
        lines.append(signal.resample(lines3[ind], samps));
        lines.append(signal.resample(lines4[ind], samps));
        #if (len(lines)==0):
        #    lines.append(signal.resample(lines2[ind], samps));
        #else:
        #    lines.append(signal.resample(lines2[ind], samps));
        #lines2[ind] = signal.resample(lines2[ind], samps);
        #print(samps);
    #print(lines.shape)
        #print(len(lines2[ind].resample(lines2[ind], samps)));
        #lines[ind] = block_mean(a, 5).shape  # (20, 40)

    testData.append(lines)
    #trainData.append(np.transpose(np.load(A_train_path)));

    N_train_path_mar = TRAIN_FILES[0] + "N_TXT/%dNmar.txt" % inde;
    lines2 = np.transpose(np.loadtxt(N_train_path_mar, comments="#", delimiter="\t", unpack=False))
    N_train_path_pie = TRAIN_FILES[0] + "N_TXT/%dNpie.txt" % inde;
    lines3 = np.transpose(np.loadtxt(N_train_path_pie, comments="#", delimiter="\t", unpack=False))
    N_train_path_sen = TRAIN_FILES[0] + "N_TXT/%dNsen.txt" % inde;
    lines4 = np.transpose(np.loadtxt(N_train_path_sen, comments="#", delimiter="\t", unpack=False))
    #lines = copy.deepcopy(lines2);
    lines = [];
    for ind, a in enumerate(lines2):
        secs = round(len(lines2[ind]) / 2000.0);  # Number of seconds in signal X
        samps = secs * 75;  # Number of samples to downsample
        #print(samps)
        #samps = 100;
        #lines2[ind] = signal.resample(lines2[ind], samps);
        lines.append(signal.resample(lines2[ind], samps));
        lines.append(signal.resample(lines3[ind], samps));
        lines.append(signal.resample(lines4[ind], samps));
        #if (len(lines)==0):
        #    lines.append(signal.resample(lines2[ind], samps));
        #else:
        #    lines.append(signal.resample(lines2[ind], samps));
        #lines2[ind] = signal.resample(lines2[ind], samps);
        #print(samps);
    #print(lines2.shape)
        #lines[ind] = block_mean(a, 5).shape  # (20, 40)

    testData.append(lines)
    DataPCA.append(lines)
    #print(lines)
    #trainData.append(np.transpose(np.load(N_train_path)));
#print(len(DataPCA))
#print(len(DataPCA[0]))
#print(len(DataPCA[0][0]))
#DataPCA = trainData;
#DataPCA.append(testData);

PDa = classLegs.PCA(DataPCA);

classLegs.predict(trainData, testData, 5)

pred = classLegs.get_preds();
for pre in pred:
    print("Total score for %d " % (pre))

print(PDa)