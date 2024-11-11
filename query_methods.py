"""
The file containing implementations to all of the query strategies. References to all of these methods can be found in
the blog that accompanies this code.
"""

import gc
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings("ignore")
from keras.models import Model
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Lambda
from keras import optimizers
from cleverhans.attacks import FastGradientMethod, DeepFool
from cleverhans.utils_keras import KerasModelWrapper

from models import *
import tensorflow as tf
import keras

import os

def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, model, input_shape=(28,28), num_labels=10, gpu=1):
        self.model = model
        self.input_shape = input_shape
        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, X_train, Y_train, labeled_idx, amount):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param X_train: the training set
        :param Y_train: the training labels
        :param labeled_idx: the indices of the labeled examples
        :param amount: the amount of examples to query
        :return: the new labeled indices (including the ones queried)
        """
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model


class RandomSampling(QueryMethod):
    """
    A random sampling query strategy baseline.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))


class UncertaintySampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        predictions = []
        step = unlabeled_idx.shape[0] // 50
        for i in range(step):
            if i != step - 1:
                predictions.extend(self.model.predict(X_train[unlabeled_idx[i * step:(i + 1) * step], :]))
            else:
                predictions.extend(
                    self.model.predict(X_train[unlabeled_idx[i * step:], :]))
        #predictions = self.model.predict(X_train[unlabeled_idx, :])

        unlabeled_predictions = np.amax(predictions, axis=1)
        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]

        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))



class UncertaintyEntropySampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the top entropy.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        #predictions = self.model.predict(X_train[unlabeled_idx, :])
        predictions = []
        step = unlabeled_idx.shape[0]//50
        for i in range(step):
            if i != step - 1:
                predictions.extend(self.model.predict(X_train[unlabeled_idx[i * step:(i + 1) * step], :]))
            else:
                predictions.extend(
                    self.model.predict(X_train[unlabeled_idx[i * step:], :]))
        predictions = np.array(predictions)
        unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)

        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))

class UncertaintyMarginSampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the top entropy.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        #predictions = self.model.predict(X_train[unlabeled_idx, :])
        predictions = []
        step = unlabeled_idx.shape[0]//50
        for i in range(step):
            if i != step - 1:
                predictions.extend(self.model.predict(X_train[unlabeled_idx[i * step:(i + 1) * step], :]))
            else:
                predictions.extend(
                    self.model.predict(X_train[unlabeled_idx[i * step:], :]))
        predictions = np.log(predictions)
        One_Two = np.argpartition(predictions, 8, axis=1)[:,-2:]

        margin = []
        for i in range(One_Two.shape[0]):
            margin.append(abs(predictions[i,One_Two[i,1]] - predictions[i,One_Two[i,0]]))
        #margin = predictions[:,One_Two[1]] - predictions[:,One_Two[0]]
        selected_indices = np.argpartition(margin, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))

