import os
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import strlearn.streams
from ..ensembles import StreamingEnsemble
from ..metrics import balanced_accuracy_score

TIME_FORMAT = '%H:%M:%S'


class Logger:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def header(message):
        return Logger.prepare_message(Logger.HEADER, message)

    @staticmethod
    def ok_green(message):
        return Logger.prepare_message(Logger.OKGREEN, message)

    @staticmethod
    def ok_cyan(message):
        return Logger.prepare_message(Logger.OKCYAN, message)

    @staticmethod
    def ok_blue(message):
        return Logger.prepare_message(Logger.OKBLUE, message)

    @staticmethod
    def fail(message):
        return Logger.prepare_message(Logger.FAIL, message)

    @staticmethod
    def prepare_message(code, message):
        return f'{code}{message}{Logger.ENDC}'

    @staticmethod
    def prefix():
        return f'[{os.getpid()}] :: {datetime.now().strftime(TIME_FORMAT)}'

    @staticmethod
    def info(message):
        print(f'{Logger.prefix()} :: {Logger.header("INFO")} :: {message}')

    @staticmethod
    def end(message):
        print(f'{Logger.prefix()} :: {Logger.ok_green("END")} :: {message}')

    @staticmethod
    def process(message):
        print(f'{Logger.prefix()} :: {Logger.ok_cyan("PROCESS")} :: {message}')

    @staticmethod
    def start(message):
        print(f'{Logger.prefix()} :: {Logger.ok_blue("START")} :: {message}')

    @staticmethod
    def error(message):
        print(f'{Logger.fail("[X]")} :: {Logger.prefix()} :: {Logger.fail("ERROR")} :: {message}')


def _make_relationship_table(d_pred):
    rt = np.zeros(shape=(2, 2))
    for idx, val in zip(*np.unique(d_pred.T, axis=0, return_counts=True)):
        rt[tuple(idx)] = val
    return rt


def make_relationship_tables(predictions, full_matrix=False):
    pool_len = len(predictions)
    if full_matrix:
        return np.array([
            [
                _make_relationship_table(predictions[(i, k), :])
                for k in range(pool_len)
            ]
            for i in range(pool_len)
        ]).T

    return np.array([
        _make_relationship_table(predictions[(i, k), :])
        for i in range(pool_len)
        for k in range(i + 1, pool_len)
    ]).T


# Q-statistic
def Q_statistic(relationship_tables):
    (n00, n01), (n10, n11) = relationship_tables
    divisor = ((n11 * n00) + (n01 * n10))
    divisor[divisor == 0] = 0.000000000001
    return ((n11 * n00) - (n01 * n10)) / divisor


# Correlation coefficient
def correlation_coefficient(relationship_tables):
    (n00, n01), (n10, n11) = relationship_tables
    return ((n11 * n00) - (n01 * n10)) / np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))


# Disagreement measure
def disagreement_measure(relationship_tables):
    (n00, n01), (n10, n11) = relationship_tables
    # return (n01 + n10) / (n11 + n10 + n01 + n00)
    return (n01 + n10) / relationship_tables.sum(axis=(0, 1))


# Double-fault measure
def double_fault_measure(relationship_tables):
    (n00, n01), (n10, n11) = relationship_tables
    # return n00 / (n11 + n10 + n01 + n00)
    return n00 / relationship_tables.sum(axis=(0, 1))


# Calculates all 5 metrics
def calc_diversity_measures(X, y, classifier_pool, p=2):
    L = len(classifier_pool)
    predictions = np.array([np.equal(_.predict(X), y).astype(np.int32) for _ in classifier_pool])
    tables = make_relationship_tables(predictions)

    q = Q_statistic(tables).mean()
    dis = disagreement_measure(tables).mean()
    kw = ((L - 1) / (2 * L)) * dis
    k = 1 - (1 / (2 * p * (1 - p))) * dis
    ensemble_predictions = np.array([member_clf.predict(X) for member_clf in classifier_pool])
    e = np.mean((L // 2 - np.abs(np.sum(y[np.newaxis, :] == ensemble_predictions, axis=0) - L // 2)) / (L / 2))
    return e, k, kw, dis, q


class TestThenTrainDiversity:
    def __init__(self, verbose=False):
        self.verbose = verbose
        warnings.filterwarnings("ignore")

    def process(self, stream: strlearn.streams.StreamGenerator,
                clfs: list[StreamingEnsemble] | tuple[StreamingEnsemble, ...]):
        """
        Perform learning procedure on data stream.

        :param stream: Data stream as an object
        :type stream: strlearn.streams.StreamGenerator
        :param clfs: scikit-learn estimator of list of scikit-learn estimators.
        :type clfs: tuple or function
        """
        # Verify if pool of classifiers or one
        self.clfs_ = clfs

        # Assign parameters
        self.stream_ = stream

        # Prepare scores table
        self.scores = np.zeros(
            (len(self.clfs_), (self.stream_.n_chunks - 1), 5)
        )

        i = 0
        sum_time = 0
        if self.verbose:
            start_time = datetime.now()
            Logger.start(f"starting at {start_time.strftime(TIME_FORMAT)}")
        while True:
            X, y = stream.get_chunk()

            start_process = time.time()
            # Test
            if stream.previous_chunk is not None:
                for clfid, clf in enumerate(self.clfs_):
                    self.scores[clfid, stream.chunk_id - 1] = np.array(calc_diversity_measures(X, y, clf.ensemble_))

            # Train
            [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]
            if self.verbose:
                i += 1
                remaining = self.stream_.n_chunks - i
                current_time = time.time() - start_process
                sum_time += current_time
                avg_time = sum_time / i
                estimated_end = datetime.now() + timedelta(seconds=remaining * avg_time)
                Logger.process(f"processed chunk {i}/{self.stream_.n_chunks} "
                               f"({current_time:.3f}s) :: estimated end {estimated_end.strftime(TIME_FORMAT)}")

            if stream.is_dry():
                if self.verbose:
                    Logger.end(f"finished at {datetime.now().strftime(TIME_FORMAT)}")
                break
