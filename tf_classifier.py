from sklearn.metrics import classification_report
import matplotlib.pylab as plt
import numpy as np
from scipy import ndimage


class ts_classifier(object):

    def __init__(self, plotter=False):
        '''
        preds is a list of predictions that will be made.
        plotter indicates whether to plot each nearest neighbor as it is found.
        '''
        self.preds = []
        self.plotter = plotter

    def predict(self, train, test, w, progress=False):
        '''
        1-nearest neighbor classification algorithm using LB_Keogh lower
        bound as similarity measure. Option to use DTW distance instead
        but is much slower.
        '''


        for ind1, i1 in enumerate(test):
            dist = [0] * len(train);
            for ind, i in enumerate(i1):
                if progress:
                    print
                    str(ind + 1) + ' points classified'
                min_dist = float('inf')
                closest_seq = []

                for j1 in range(len(train)):
                    #print(j1)
                    #print(test)
                    #print("whaaaa")
                    #print(train)
                    #print("hmm.....")
                    #print(len(i1))
                    #print(len(train[j1]))
                    #print(ind)
                    #print(train[j1][ind][:-1])
                    dist[j1] = dist[j1]+np.square(self.LB_Keogh(i, train[j1][ind][:-1], 5))

                    #if dist[j1] < min_dist:
                    #    dist2 = self.DTWDistance(i, train[dis][ind][:-1], w);
                    #    if dist2 < min_dist:
                    #        min_dist = dist[j1];
                    #        closest_seq = train[dis][ind]
                    #self.preds.append(closest_seq[-1])
                    #print("weeeeell!!")
                    #print(dist[j1])
            for dis in range(len(dist)):
                dist2 = 0;
                #min_dist = float('inf');
                if np.sqrt(dist[dis]) < min_dist:
                    for ind, i in enumerate(i1):
                        #print(train[dis].shape)
                        dist2 = dist2+np.square(self.DTWDistance(i, train[dis][ind][:-1], w))
                    if np.sqrt(dist2) < min_dist:
                        min_dist = np.sqrt(dist2)
                        closest_seq = train[dis][0]
                        print("ind")
                        print(dis)
            #print("pred seq")
            #print(closest_seq)
            self.preds.append(closest_seq[-1])

            if self.plotter:
                plt.plot(i)
                plt.plot(closest_seq[:-1])
                plt.legend(['Test Series', 'Nearest Neighbor in Training Set'])
                plt.title('Nearest Neighbor in Training Set - Prediction =' + str(closest_seq[-1]))
                plt.show()

    def PCA(self, data):
        #var = [0] * len(data);
        AveSeq = [];
        varList = [];
        for indexDim in range(len(data[0])):
            #print("len")
            #print(len(data[0]))
            dataAdd = [];
            disSeq = [];
            for ind1, i1 in enumerate(data):
                #print(i1[indexDim])
                #print(len(data))
                #print(len(i1))
                #print(len(i1[0]))
                dataAdd.append(i1[indexDim]);
            AveSeq.append(self.LB_Average(dataAdd, 5));
            for ind1, i1 in enumerate(data):
                disSeq.append(self.LB_Keogh(AveSeq[indexDim], i1[indexDim], 5));
            disSeq = [float(i)/max(disSeq) for i in disSeq]
            varList.append(np.var(disSeq));
        return varList




    def performance(self, true_results):
        '''
        If the actual test set labels are known, can determine classification
        accuracy.
        '''
        return classification_report(true_results, self.preds)

    def get_preds(self):
        return self.preds

    def DTWDistance(self, s1, s2, w=None):
        '''
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        '''
        DTW = {}

        if w:
            w = max(w, abs(len(s1) - len(s2)))

            for i in range(-1, len(s1)):
                for j in range(-1, len(s2)):
                    #print("ooohhhh")
                    DTW[(i, j)] = float('inf')

        else:
            for i in range(len(s1)):
                #print("whaaattt")
                #print(str(len(s1)));
                DTW[(i, -1)] = float('inf')
            for i in range(len(s2)):
                DTW[(-1, i)] = float('inf')
        #print("done 1")
        DTW[(-1, -1)] = 0
        #print(str(len(DTW)))
        for i in range(len(s1)):
            if w:
                for j in range(max(0, i - w), min(len(s2), i + w)):
                    dist = (s1[i] - s2[j]) ** 2
                    DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
            else:
                for j in range(len(s2)):
                    #print("hoooowwww???")
                    dist = (s1[i] - s2[j]) ** 2
                    DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
        #print("hehehehe")
        #print(DTW)

        return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])
    def LB_Average(self, VecSeq, r):
        '''
                Calculates LB_Keough lower bound to dynamic time warping. Linear
                complexity compared to quadratic complexity of dtw.
                '''
        LB_sum = 0
        # for ind, i in enumerate(s1):
        disAv = [];
        for i, seq in enumerate(VecSeq):
            #disAv[i] = np.mean(np.array([disAv[i],)
            for ind in range(len(seq)):
                # print(ind)
                # print(i)
                # print(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

                # print(ind-r)
                #print(seq[0])
                lower_bound = min(seq[(ind - r if ind - r >= 0 else 0):(ind + r)]);
                upper_bound = max(seq[(ind - r if ind - r >= 0 else 0):(ind + r)]);
                if(len(disAv)>ind):
                    #print(len(disAv))
                    #print(ind)
                    disAv[ind] = np.mean([np.mean([lower_bound,upper_bound]), disAv[ind]]);
                else:
                    disAv.append(np.mean([lower_bound,upper_bound]));

                # if i > upper_bound:
                # print(type(s2[0]))
                #if s1[ind] > upper_bound:
                    # LB_sum = LB_sum + (i - upper_bound) ** 2
                #    LB_sum = LB_sum + (s1[ind] - upper_bound) ** 2
                # elif i < lower_bound:
                #elif s1[ind] < lower_bound:
                    # LB_sum = LB_sum + (i - lower_bound) ** 2
                #    LB_sum = LB_sum + (s1[ind] - lower_bound) ** 2

        return disAv

    def LB_Keogh(self, s1, s2, r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum = 0
        #print(s1)
        #print(s2)
        #for ind, i in enumerate(s1):
        for ind in range(min(len(s1),len(s2))):
            #print(ind)
            #print(i)
            #print(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

            #print(ind-r)
            lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
            upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

            #if i > upper_bound:
            #print(type(s2[0]))
            if s1[ind] > upper_bound:
                #LB_sum = LB_sum + (i - upper_bound) ** 2
                LB_sum = LB_sum + (s1[ind] - upper_bound) ** 2
            #elif i < lower_bound:
            elif s1[ind] < lower_bound:
                #LB_sum = LB_sum + (i - lower_bound) ** 2
                LB_sum = LB_sum + (s1[ind] - lower_bound) ** 2

        return np.sqrt(LB_sum)

def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res