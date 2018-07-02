import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import scipy.io as sio
from sklearn.preprocessing import MultiLabelBinarizer
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)  
    from statsmodels.tsa.arima_process import ArmaProcess
    from statsmodels.tsa.arima_model import ARMA

# Utilities for ACM Athens Data Science Summer School: Time Series.
# Alex Bird June 2018. Some functions copied from github.com/ornithos/pyalexutil

MM = os.path.splitext(os.path.basename(__file__))[0]

def _run_length_encoding(x):
    """
    from https://github.com/ornithos/pyalexutil/manipulate.py
    For 1D array x, turn [0,0,0,0,1,1,1,1,0,1,1,1] into [0, 3, 7, 8], 
    [3, 4, 1, 4], [0, 1, 0, 1]
    
    :param x:
    :return: (indices of changes, length of runs, new number (assuming boolean).
    """
    x = np.asarray(x)
    assert x.ndim == 1, "run_length_encoding currently only supports 1D arrays"

    changes = x[:-1] != x[1:]
    changes_ix = np.where(changes)[0] + 1
    changes_from = np.concatenate(([int(not x[0])], x[changes_ix]))
    changes_ix = np.concatenate(([0], changes_ix))
    changes_to = np.logical_not(changes_from).astype(int)
    lengths = np.diff(np.concatenate((changes_ix, [x.size])))
    return changes_ix, lengths, changes_to


def _splt_gridsize(num):
    """ 
    Calculates a grid size for the number of plots specified s.t. no space is wasted, and is close to sq.
    Shamelessely stolen from Matt Johnson - https://github.com/mattjj/pymattutil/plot.py
    """
    return sorted(min([(x,int(np.ceil(num/x))) for x in range(1,int(np.floor(np.sqrt(num)))+1)],key=sum))


class mhealth_data(object):
    """
    Helper class to load and access data / annotations. Only the defined
    subset and spliced file is used here.

    initialise: obj = mhealth_data(fpath='mhealth-athens.csv')

    methods:
        * obj.data               -- return observation DataFrame.
        * obj.annotation(classnm)-- return boolean Series of req. class name (or id)
        * obj.annotations()      -- return Series of all class ids. Available as one-hot.
        * obj.intervals(classnm) -- return matrix of annotation intervals of req. type.
        * obj.chunks(classnm)    -- return list of chunks of observations corresponding
                                     to patient id and annotation type.

    Note that in all cases 'type' includes the annotation 'Normal' for the unmarked
    periods.
    """
    
    def __init__(self, fpath='mhealth-athens.csv'):
        data = pd.read_csv(fpath)
        n = len(data)
        annotations = data['class']
        data = data.drop('class', axis=1)
        data.columns = np.arange(5) + 1

        intervals_mhealth = []
        for i in range(1, 6):
            tmp_marked = (annotations.values == i) * 1
            norm_start, norm_len, norm_type = _run_length_encoding(tmp_marked)
            pos_start = (norm_type[0] == 0).astype('int')
            interval_cur = np.stack((norm_start[pos_start::2],
                                    norm_len[pos_start::2])).T
            interval_cur[:, 1] += interval_cur[:, 0]
            intervals_mhealth.append(interval_cur)

        self.data = data
        self.n = n
        self.k = 4
        self._annotations = annotations
        self._intervals = intervals_mhealth
        self._annotation_types = ['Standing', 'Walking', 'Climbing Stairs', 'Running']
        
    def _lkp_annotation_type(self, classnm):
        """
        (internal) lookup annotation class name to column index.
        """
        if isinstance(classnm, int):
            return classnm
        elif isinstance(classnm, str):
            assert classnm in self._annotation_types, "Must choose classnm in " + \
                               "{:s}".format(", ".join(self._annotation_types))
            
            return self._annotation_types.index(classnm) + 1
        else:
            raise RuntimeError("Unexpected class requested. Expecting int or str.")
    
    def annotation(self, classnm):
        """
        :param classnm: activity type. Can be numeric or str with class name.
        :return: boolean Series marked 'True' if requested type annotated at
        each timepoint 0...86400.
        """
        classid = self._lkp_annotation_type(classnm)
        return self._annotations == classnm
    
    def annotations(self, as_one_hot=False):
        """
        :param as_one_hot: (default False): annotations returned as numeric Series.
                            (True): returnd a DataFrame with each class a one-hot column.
        :return: all class annotations at each time point. The one-hot representation is
        useful for plotting (.annotations(as_one_hot=True).plot()).
        """
        if not as_one_hot:
            return self._annotations
        else:
            lblbin = MultiLabelBinarizer().fit_transform(self._annotations.values.reshape(-1,1))
            return pd.DataFrame(lblbin, columns=np.arange(self.k) + 1)
        
    def intervals(self, classnm):
        """
        :param classnm: activity type. Can be numeric or str with class name.
        :return: m x 2 matrix of interval indices for a given annotation
        for id in [1, ..., 4].
        """
        classid = self._lkp_annotation_type(classnm)
        return self._intervals[classid-1]
    
    def chunks(self, classnm):
        """
        :param classnm: activity type. Can be numeric or str with class name.
        :return: a list of the chunks of time series of class 'classnm'
        """
        intervals = self.intervals(classnm)
        out = []
        for ixs in intervals:
            if ixs.size == 0:
                break
            out.append(self.data[slice(*ixs)])
        return out

    def training(self, amt=0.7):
        """
        :param amt: (optional) proportion of dataset to use (default=0.7)
        :return: X, y -- both DataFrames corresponding to .data / .annotations(one_hot=True)
        taking the first amt * n values (time series so not shuffled.)
        """
        trn_n = np.round(self.n * 0.7)
        y = self.annotations(as_one_hot=False).loc[:trn_n]
        return self.data.loc[:trn_n, :], y

    def test(self, trn_amt=0.7):
        """
        :param trn_amt: (optional) proportion of dataset to use *for training* (default=0.7)
        :return: X, y -- both DataFrames corresponding to .data / .annotations(one_hot=True)
        taking the values after amt * n values used for training
        """
        trn_n = np.round(self.n * 0.7)
        y = self.annotations(as_one_hot=False).loc[trn_n:]
        return self.data.loc[trn_n::], y


class SingletonLoadMhealth(object):
    __instance = None
    def __new__(cls):
        if SingletonLoadMhealth.__instance is None:
            SingletonLoadMhealth.__instance = object.__new__(cls)
            SingletonLoadMhealth.__instance.data = mhealth_data()
        return SingletonLoadMhealth.__instance.data


def _get_stairs_ar_prediction_data(training_set, test_set, training_ixs):
    trn_0 = training_set is None
    tst_0 = test_set is None
    ixs_0 = training_ixs is None
    assert trn_0 == tst_0, "either both training and test must be specified or neither"
    assert ixs_0 == tst_0, "either all {training, test, ixs} must be specified or none."

    if trn_0:
        mhdata = SingletonLoadMhealth()
        training_ixs = [100,200,300,400,500,600,700,800,900]
        training_set = [mhdata.chunks('Climbing Stairs')[2][:ix] for ix in training_ixs]
        test_set = [mhdata.chunks('Climbing Stairs')[2][ix:(ix+60)] for ix in training_ixs]
    
    return training_set, test_set, training_ixs


def ARMA_plot_predictions(arma_order, channel, training_set=None, test_set=None, training_ixs=None, title=None):
    """
    plot predictions for Exercise 1 for specified ARMA order.
    """
    assert isinstance(arma_order, (tuple, list)) and len(arma_order) == 2 , "arma_oder must be a len 2 list."
    assert isinstance(channel, int) and 0 < channel <= 6, "channel should be an int 1,2,3,4,5."
    training_set, test_set, training_ixs = _get_stairs_ar_prediction_data(training_set, test_set, training_ixs)
    n = len(training_set)
    assert len(test_set) == n, "training set and test set of different lengths."
    assert len(training_ixs) == n, "training set and training_ixs of different lengths."
    
    f, axs = plt.subplots(*_splt_gridsize(n))
    axs = axs.ravel()
    for trn, tst, i, ax in zip(training_set, test_set, training_ixs, axs):
        ar_model = ARMA(trn[channel].values, order=arma_order)
        ar_model = ar_model.fit(method='css')
        forecast = ar_model.predict(start=i, end=i+59)
        ax.plot(np.arange(i-60, i), trn[channel][-60:])
        ax.plot(np.arange(i, i+60), tst[channel].values)
        ax.plot(np.arange(i, i+60), forecast)
        ax.axvline(i, color='k', linestyle=':')
    f.set_size_inches(8,6)
    if title:
        f.suptitle(title)
    plt.tight_layout()


def ARMA_mse_predictions(arma_order, channel, training_set=None, test_set=None, training_ixs=None, step_ahead=60):
    """
    get MSE of predictions for Exercise 1 for specified ARMA order.
    """
    assert isinstance(arma_order, (tuple, list)) and len(arma_order) == 2 , "arma_oder must be a len 2 list."
    assert isinstance(channel, int) and 0 < channel <= 6, "channel should be an int 1,2,3,4,5."
    training_set, test_set, training_ixs = _get_stairs_ar_prediction_data(training_set, test_set, training_ixs)
    n = len(training_set)
    assert len(test_set) == n, "training set and test set of different lengths."
    assert len(training_ixs) == n, "training set and training_ixs of different lengths."

    mses = []
    for trn, tst, i in zip(training_set, test_set, training_ixs):
        ar_model = ARMA(trn[channel].values, order=arma_order)
        ar_model = ar_model.fit(method='css')
        forecast = ar_model.predict(start=i, end=i+step_ahead-1)

        actual = tst[channel].values[:step_ahead]
        mses.append(np.mean((actual-forecast)**2))

    return mses


def check_training_data(x1, x2, x3, x4, x5):
    # ----------------------------
    # NO CHEATING!!!
    # NEED TO ENCRPYT THIS IF I HAVE TIME
    # -----------------------------

    mhdata = SingletonLoadMhealth()
    X, y = mhdata.training()

    n, d = X.shape
    Xtrain_wdws = []

    for k, x in X.iteritems():
        tmp = np.zeros((n-5, 6))
        for i in range(n-5):
            tmp[i,:] = x.values[i:i+6]
        Xtrain_wdws.append(tmp)

    def check_x(x_check, x_true, id):
        assert x_check.shape == x_true.shape, "x{:d} is of shape {:s}. Expecting {:s}".format(id, x_check.shape.__str__(), x_true.shape.__str__())
        if not np.allclose(x_check, x_true):
            tmp = np.isclose(x_check, x_true)
            first_index = np.array(np.where(~tmp))[:,0]
            if id == 5 and all(x==y for x,y in zip(first_index, (np.array(x_check.shape) - 1))):
                raise RuntimeError("Cheating Detected!")
            print("x{:d} is not right. First incorrect index:".format(id))
            print("{:s}".format(first_index.__str__()))
            raise RuntimeError("Incorrect x{:d} given".format(id))

    check_x(x1, Xtrain_wdws[0], 1)
    check_x(x2, Xtrain_wdws[1], 2)
    check_x(x3, Xtrain_wdws[2], 3)
    check_x(x4, Xtrain_wdws[3], 4)
    check_x(x5, Xtrain_wdws[4], 5)

    key = MM
    code = '¤ãÔÔàßÙæè\x94ÐÞÆÙÖ\x85××ÙÔè\x94ÜØÆÙØ\x85ÔèæÜäéÜØÚ¢'
    
    encoded_chars = []
    for i in range(len(code)):
        key_c = key[i % len(key)]
        encoded_c = chr(ord(code[i]) - ord(key_c) % 256)
        encoded_chars.append(encoded_c)
    passphrase = "".join(encoded_chars)
    
    print(" ------------------ CORRECT --------------------")
    print("CODEPHRASE FOR EXERCISE: " + passphrase)


def get_sliding_training_test_data(as_lists=True):
    """
     Return training and test data in sliding window dataframes.
     :param as_lists (Default True)
     [FALSE]: Will return Xtrain, Xtest as wide DataFrames where the 20-step
     sliding window for each channel is concatenated horizontally,
     [TRUE]: Returns Xtrain, Xtest as lists of DataFrames where each element
     corresponds to a channel.
     Output: Xtrain, ytrain, Xtest, ytest
    """

    # ----------------------------
    # NO CHEATING!!!
    # NEED TO ENCRPYT THIS IF I HAVE TIME
    # -----------------------------
    mhdata = SingletonLoadMhealth()
    X, ytrain_ = mhdata.training()
    Xtest, ytest_ = mhdata.test()

    n, d = X.shape
    ytrain = ytrain_[5:]
    Xtrain_wdws = []

    for k, x in X.iteritems():
        tmp = np.zeros((n-5, 6))
        for i in range(n-5):
            tmp[i,:] = x.values[i:i+6]
        Xtrain_wdws.append(tmp)

    Xtrain_wdws[4][-1,-1] += 1e-2  # avoid cheats

    ntest, dtest = Xtest.shape
    ytest = ytest_[5:]
    Xtest_wdws = []

    for k, x in Xtest.iteritems():
        tmp = np.zeros((ntest-5, 6))
        for i in range(ntest-5):
            tmp[i,:] = x.values[i:i+6]
        Xtest_wdws.append(tmp)

    if as_lists:
        Xtrain = Xtrain_wdws
        Xtest = Xtest_wdws
    else:
        Xtrain = pd.DataFrame(np.hstack(Xtrain_wdws))
        Xtest = pd.DataFrame(np.hstack(Xtest_wdws))

    return Xtrain, ytrain, Xtest, ytest


# def _load_baby_data(fpath):
#     # Ingest MATLAB data file .mat, collect into relevant numpy
#     # arrays per individuals alongside one-hot matrices of events.
#     #
#     # can extract struct fieldnames from matlab struct by using
#     # option struct_as_record=False, and use hidden field _filednames.
#     # We don't do this here as it's a little slower, and I've pre-
#     # extracted them.
#     #
#     # .mat files are interpreted by scipy as being one single numpy array.
#     # This is a shame, since .mat files may contain many objects of different
#     # types. This makes indices a little horrible.

#     matlab_data = sio.loadmat(fpath)
#     _data = matlab_data['data']
#     _intervals = matlab_data['intervals'][0, 0]

#     interval_names = ['BloodSample', 'Bradycardia', 'CoreTempProbeDisconnect',
#                       'IncubatorOpen', 'Normal', 'TCP', 'Abnormal']

#     # Extract observed time series
#     _individuals = _data[0][0][0][0]
#     labels = [[y[0] for y in x[0][0][1][0]] for x in _individuals]
#     data = [x[0][0][6] for x in _individuals]
#     data = [pd.DataFrame(x, columns=lbl) for x, lbl in zip(data, labels)]

#     # Extract annotations, make one hot arrays.
#     intervals = []
#     intervals_one_hot = []
#     for _id in range(15):
#         matb = np.zeros((86400, 7))
#         intervals.append([_intervals[j][_id][0] for j in range(7)])
#         for j in range(7):
#             for ixs in _intervals[j][_id][0]:
#                 if ixs.size == 0:
#                     break
#                 matb[slice(*ixs), j] = 1

#         dfb = pd.DataFrame(matb == 1, columns=interval_names)
#         intervals_one_hot.append(dfb)

#     return data, intervals_one_hot, intervals


# class neonatal_data(object):
#     """
#     Helper class to load and access data / annotations. Only 10 patients are
#     returned since 5 of the original cohort did not include Blood Pressure
#     channels. In principle, the entire cohort can be retrieved using the
#     argument 'ids=range(15)' but this is unnecessary for this exercise.

#     initialise: obj = neonatal_data(filepath_to_matlab file)

#     methods:
#         * obj.data(id)            -- return observation DataFrame for selected patient id.
#         * obj.intervals(id, type) -- return matrix of annotation intervals of req. type.
#         * obj.chunks(id, type)    -- return list of chunks of observations corresponding
#                                      to patient id and annotation type.

#     Note that in all cases 'type' includes the annotation 'Normal' for the unmarked
#     periods.

#     """

#     def __init__(self, fpath='15days.mat',
#                  columns=('HR', 'TC', 'BS', 'SO', 'Incu.Air Temp'),
#                  ids=(0, 3, 4, 7, 8, 9, 11, 12, 13, 14)):
#         data, annotations, intervals = _load_baby_data(fpath)
#         n = len(ids)
#         self._data = [data[i].loc[:, columns] for i in ids]
#         self._annotations = [annotations[i] for i in ids]
#         self._intervals = [intervals[i] for i in ids]
#         self._annotation_types = self._annotations[0].columns.values.tolist()

#         # calculate normal intervals
#         intervals_unmarked = []
#         for i in range(n):
#             tmp_marked = np.any(self._annotations[i].as_matrix(), axis=1)
#             norm_start, norm_len, norm_type = _run_length_encoding(tmp_marked)
#             pos_start = (norm_type[0] == 1).astype('int')
#             interval_um = np.stack((norm_start[pos_start::2],
#                                     norm_len[pos_start::2])).T
#             interval_um[:, 1] += interval_um[:, 0]
#             intervals_unmarked.append(interval_um)

#         self._intervals_unmarked = intervals_unmarked

#     def _lkp_annotation_type(self, type):
#         """
#         (internal) lookup annotation type to column index.
#         """
#         return self._annotation_types.index(type)

#     def data(self, id):
#         """
#         :param id: patient id
#         :return: DataFrame of all observations for id in [0, ..., 9]
#         """
#         assert isinstance(id, int), "id must an integer"
#         return self._data[id]

#     def annotation(self, id, type):
#         """
#         :param id: patient id
#         :param type: annotation type. Trial and use error msg for valid types.
#         :return: boolean Series marked 'True' if requested type annotated at
#         each timepoint 0...86400.
#         """
#         assert isinstance(id, int), "id must an integer"
#         assert isinstance(type, str), "type must be a string"
#         if type in self._annotation_types:
#             return self._annotations[id][type]
#         elif type == 'Unmarked':
#             return np.logical_not(np.any(self._annotations[id].as_matrix(),
#                                          axis=1))
#         else:
#             raise RuntimeError("Must choose type in Unmarked, " + \
#                                "{:s}".format(", ".join(self._annotation_types)))

#     def annotations(self, id):
#         """
#         :param id: patient id
#         :return: 0/1 DataFrame for all aggrefated annotation types except
#         'Unmarked', useful for plotting.
#         """
#         assert isinstance(id, int), "id must an integer"
#         return self._annotations[id].apply(lambda x: x*1)


#     def intervals(self, id, type):
#         """
#         :param id: patient id
#         :param type: annotation type. Trial and use error msg for valid types.
#         :return: m x 2 matrix of interval indices for a given annotation
#         for id in [0, ..., 9].

#         Each row of this matrix denotes the start and end indices of the
#         annotation.
#         """
#         assert isinstance(id, int), "id must an integer"
#         assert isinstance(type, str), "type must be a string"
#         if type in self._annotation_types:
#             return self._intervals[id][self._lkp_annotation_type(type)]
#         elif type == 'Unmarked':
#             return self._intervals_unmarked[id]
#         else:
#             raise RuntimeError("Must choose type in Unmarked, " + \
#                                "{:s}".format(", ".join(self._annotation_types)))

#     def chunks(self, id, type):
#         """
#         :param id: patient id
#         :param type: annotation type. Trial and use error msg for valid types.
#         :return: a list of the chunks of time series from selected patient id
#         with type indicated. 
#         """
#         intervals = self.intervals(id, type)
#         assert isinstance(id, int), "id must an integer"
#         out = []
#         for ixs in intervals:
#             if ixs.size == 0:
#                 break
#             out.append(self._data[id][slice(*ixs)])
#         return out