#from keras.models import Model

#from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
#from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

#from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
#from utils.keras_utils import train_model, evaluate_model, set_trainable
#from utils.layer_utils import AttentionLSTM
from random import shuffle
import logging
import random
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import signal
from tf_classifier import *
from RandomForest import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

def test_rf2():

    from sklearn.model_selection import train_test_split
    import copy
    import matplotlib.pyplot as plt
    from random import shuffle

    classLegs = ts_classifier();
    TRAIN_FILES = ["Data/"]
    fold_index = 9;
    trainData = [];
    testData = [];
    data = []
    plot_idx = 1

    plot_step = 0.02  # fine step width for decision surface contours
    plot_step_coarser = 0.5  # step widths for coarse classifier guesses
    cmap = plt.cm.RdYlBu

    lisInd = [1,2,3,4,5,6,7,8,9,10,11];
    shuffle(lisInd);
    #print(lisInd);
    for inde1 in range(fold_index):
        #print("inde1")
        #print(inde1)
        inde = lisInd[inde1]
        #print("inde train")
        #print(inde)
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

            #lines.append(np.append(signal.resample(lines2[ind], samps),np.array([1])));
            #plt.figure(1)
            #plt.subplot(211)
            #plt.plot(a)
            #plt.subplot(212)
            #plt.plot(lines[-1][:-1])
            #plt.show()
            #lines.append(np.append(signal.resample(lines3[ind], samps),np.array([1])));
            #lines2[ind] = (lines2[ind] - lines2[ind].min()) / (lines2[ind].max() - lines2[ind].min())
            #lines.append(np.append(signal.resample(lines2[ind], samps), np.array([1])));
            #lines3[ind] = (lines3[ind] - lines3[ind].min()) / (lines3[ind].max() - lines3[ind].min())
            #lines.append(np.append(signal.resample(lines3[ind], samps), np.array([1])));
            lines4[ind] = (lines4[ind] - lines4[ind].min()) / (lines4[ind].max() - lines4[ind].min())
            #lines.append(np.append(signal.resample(lines4[ind], samps), np.array([1])));
            if(ind==3 or ind==4):
                lines.append(np.append(signal.resample(lines4[ind], samps), np.array([1])));
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
        data.append(lines)
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
            #lines.append(np.append(signal.resample(lines2[ind], samps),np.array([0])));
            #plt.figure(1)
            #plt.subplot(211)
            #plt.plot(a)
            #plt.subplot(212)
            #plt.plot(lines[-1][:-1])
            #plt.show()
            #lines.append(np.append(signal.resample(lines3[ind], samps),np.array([0])));

            #lines2[ind] = (lines2[ind] - lines2[ind].min()) / (lines2[ind].max() - lines2[ind].min())
            #lines.append(np.append(signal.resample(lines2[ind], samps), np.array([0])));
            #lines3[ind] = (lines3[ind] - lines3[ind].min()) / (lines3[ind].max() - lines3[ind].min())
            #lines.append(np.append(signal.resample(lines3[ind], samps), np.array([0])));
            lines4[ind] = (lines4[ind] - lines4[ind].min()) / (lines4[ind].max() - lines4[ind].min())
            #lines.append(np.append(signal.resample(lines4[ind], samps), np.array([0])));
            if(ind==3 or ind==4):
                lines.append(np.append(signal.resample(lines4[ind], samps), np.array([0])));
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
        data.append(lines)
        #trainData.append(np.transpose(np.load(N_train_path)));

    for inde in range(9,11):
        #print("inde1")
        #print(inde1)
        inde = lisInd[inde1]
        #A_test_path = TRAIN_FILES + "A_TXT/%dAmar.txt" % inde;
        #testData.append(np.transpose(np.load(A_test_path)));

        #N_test_path = TRAIN_FILES + "N_TXT/%dNmar.txt" % inde;
        #testData.append(np.transpose(np.load(N_test_path)));
        #print("inde test")
        #print(inde)
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
            #lines.append(signal.resample(lines2[ind], samps));
            #lines.append(signal.resample(lines3[ind], samps));
            #lines.append(np.append(signal.resample(lines3[ind], samps), np.array([1])));
            #lines2[ind] = (lines2[ind] - lines2[ind].min()) / (lines2[ind].max() - lines2[ind].min())
            #lines.append(np.append(signal.resample(lines2[ind], samps), np.array([1])));
            #lines3[ind] = (lines3[ind] - lines3[ind].min()) / (lines3[ind].max() - lines3[ind].min())
            #lines.append(np.append(signal.resample(lines3[ind], samps), np.array([1])));
            lines4[ind] = (lines4[ind] - lines4[ind].min()) / (lines4[ind].max() - lines4[ind].min())
            #lines.append(np.append(signal.resample(lines4[ind], samps), np.array([1])));
            if(ind==3 or ind==4):
                lines.append(np.append(signal.resample(lines4[ind], samps), np.array([1])));
            #lines.append(signal.resample(lines4[ind], samps));
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
        data.append(lines)
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
            #print(samps)
            #samps = 100;
            #lines2[ind] = signal.resample(lines2[in
            # d], samps);
            #lines.append(signal.resample(lines2[ind], samps));
            #lines.append(signal.resample(lines3[ind], samps));
            #lines.append(np.append(signal.resample(lines3[ind], samps), np.array([0])));

            #lines2[ind] = (lines2[ind] - lines2[ind].min()) / (lines2[ind].max() - lines2[ind].min())
            #lines.append(np.append(signal.resample(lines2[ind], samps), np.array([0])));
            #lines3[ind] = (lines3[ind] - lines3[ind].min()) / (lines3[ind].max() - lines3[ind].min())
            #lines.append(np.append(signal.resample(lines3[ind], samps), np.array([0])));
            lines4[ind] = (lines4[ind] - lines4[ind].min()) / (lines4[ind].max() - lines4[ind].min())
            #lines.append(np.append(signal.resample(lines4[ind], samps), np.array([0])));
            if(ind==3 or ind==4):
                lines.append(np.append(signal.resample(lines4[ind], samps), np.array([0])));
            #lines.append(signal.resample(lines4[ind], samps));
            #if (len(lines)==0):
            #    lines.append(signal.resample(lines2[ind], samps));
            #else:
            #    lines.append(signal.resample(lines2[ind], samps));
            #lines2[ind] = signal.resample(lines2[ind], samps);
            #print(samps);
        #print(lines2.shape)
            #lines[ind] = block_mean(a, 5).shape  # (20, 40)

        testData.append(lines);
        data.append(lines);
        #trainData.append(np.transpose(np.load(N_train_path)));
    #print(len(trainData))
    #trainData.append(testData);
    #print("hmmm")
    #print(len(trainData))
    #print(len(testData))
    #print(len(data))
    distrain = classLegs.disExtractDim(data,10);
    #print("what")
    #print(distrain)
    #distest = classLegs.disExtractTest(trainData,10);
    shuffle(distrain)
    #RandLegs = RandomForestClassifier(1, 10);
    train, test = train_test_split(distrain, test_size=0.22)
    X = [];
    y = [];
    for cre in range(len(train)):
        X.append(train[cre][0])
        y.append(train[cre][1])
    X = [ro[:-1] for ro in train]
    X = np.array(X)
    print("X shape")
    print(X.shape)
    yC = [ro[-1] for ro in train]
    yC = np.array(yC)

    #X.resize(len(train),1)
    #print(X.shape)
    #y = np.array(y)
    #y.resize(len(train),1)
    #print("huh")
    #print(train)
    #print(test)

    model = GaussianNB()
    #model = MultinomialNB()
    #model = BernoulliNB()
    model.fit(X, yC);

    #for cre in range(len(train)):
    #    Xnew.append(test[cre][0])
    #    ynewVal.append(test[cre][1])
    Xnew = [ft[:-1] for ft in test]
    ynewVal = [ft[-1] for ft in test]
    Xnew = np.array(Xnew)
    #Xnew.resize(len(test), 1)
    ynewVal = np.array(ynewVal)
    #ynewVal.resize(len(test), 1)

    #print("Xnew")
    #print(Xnew[0].shape)
    ynew = model.predict(Xnew)
    #rf = RandomForestClassifier(nb_trees=60, nb_samples=5, max_workers=4)
    #print(type(train))
    #rf.fit(list(train))

    errors = 0
    #features = [ft[:-1] for ft in test]
    #values = [ft[-1] for ft in test]

    for prediction, value in zip(ynew, ynewVal):
        #prediction = rf.predict(feature)
        print("okay")
        print(prediction)
        print(value)
        if prediction != value:
            errors += 1
    logging.info("Error rate: {}".format(errors / len(ynew) * 100))
    tn, fp, fn, tp = confusion_matrix(ynewVal, ynew).ravel()
    print(tn)
    print(fp)
    print(fn)
    print(tp)

    if(len(trainData[0])==2):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        #xx_coarser, yy_coarser = np.meshgrid(
        #    np.arange(x_min, x_max, plot_step_coarser),
        #    np.arange(y_min, y_max, plot_step_coarser))
        #print("hmm")
        #print(y_min)
        #print(y_max)

        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / 1 #len(model.trees)

            #print("tree")
            #print(xx.shape)
            #print(type(xx))
            #print(type(np.ravel(xx)))
            xxtest = np.ravel(xx)
            #xxtest.reshape(len(xx.ravel()), 1)
            yytest = np.ravel(yy)
            #yytest.reshape(len(yy.ravel()), 1)
            #Z = []
            #for x, y in zip(xxtest, yytest):
            #print(np.c_[xx.ravel, yy.ravel].shape.reshape(2,1))
            #    print("x")
            #    print(x)
            #    Z.append(model.predict([x, y]))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            #Z.append(tree.predict(list(np.raxx)))
            #Z = np.array(Z)
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)


        # Plot either
        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        xxtest = np.ravel(xx_coarser)
        yytest = np.ravel(yy_coarser)
        #print(yy_coarser.ravel())
        #Z_points_coarser = []
        #for x, y in zip(xxtest, yytest):
            # print(np.c_[xx.ravel, yy.ravel].shape.reshape(2,1))
        #    Z_points_coarser.append(model.predict([x, y]))

        #Z_points_coarser = np.array(Z_points_coarser)

        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        Z_points_coarser = Z_points_coarser.reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline

        plt.scatter(X[:, 0], X[:, 1], c=yC,
                    cmap=ListedColormap(['r', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

        #export_graphviz(rf.estimators_,
        #                filled=True,
        #                rounded=True)
        #os.system('dot -Tpng tree.dot -o tree.png')
        #pred = classLegs.get_preds();
        #for pre in pred:
        #    print("Total score for %d " % (pre))
        plt.suptitle("Classifiers on feature subsets of the Iris dataset")
        plt.axis("tight")

        plt.show()
    #pred = classLegs.get_preds();
    #for pre in pred:
    #    print("Total score for %d " % (pre))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_rf2()