import multiprocessing
import os.path
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import freeze_support

import numpy as np
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from strlearn.ensembles import ONSBoost, UOB, RandomClassifier, OnlineBoosting, OOB, OnlineBagging
from strlearn.evaluators import TestThenTrain
from strlearn.streams import StreamGenerator, NPYParser
from strlearn.metrics import balanced_accuracy_score, f1_score, recall

import warnings

SAVED_STREAMS_PATH = "D:\\JavaTraining\\mgr\\saved_streams"
SAVED_SCORES_PATH = "D:\\JavaTraining\\mgr\\saved_scores"
CDI_PATH = 'cdi'
DDI_PATH = 'ddi'
DISCO_PATH = 'disco'

N_DRIFTS = 3
CONCEPT_SIGMOID_SPACING = 5
N_CHUNKS = 250
N_FEATURES = 10
CHUNK_SIZE = 200

warnings.filterwarnings("ignore")

DDI_WEIGHTS = [
    [.2, .8],
    [.3, .6],
    [.1, .9],
    [.2, .9]
]
CDI_WEIGHTS = [
    (2, 4, .8),
    (2, 5, .9),
    (3, 4, .8),
    (3, 5, .9)
]


def prepare_di_stream(seed, weight):
    return StreamGenerator(weights=weight,
                           random_state=seed,
                           chunk_size=CHUNK_SIZE,
                           n_chunks=N_CHUNKS,
                           n_features=N_FEATURES), \
        f"w{weight}_s{seed}_nc{N_CHUNKS}_nf{N_FEATURES}_cs{CHUNK_SIZE}"


def prepare_disco_stream(seed, weight):
    return StreamGenerator(weights=weight,
                           random_state=seed,
                           n_drifts=N_DRIFTS,
                           concept_sigmoid_spacing=CONCEPT_SIGMOID_SPACING,
                           recurring=True,
                           incremental=True,
                           chunk_size=CHUNK_SIZE,
                           n_chunks=N_CHUNKS,
                           n_features=N_FEATURES), \
        f"w{weight}_s{seed}_nc{N_CHUNKS}_nf{N_FEATURES}_cs{CHUNK_SIZE}"


RANDOM_STATES = [1000, 100000, 101010,
                 10110, 101101, 1001,
                 10101010, 101, 110, 1337]
BASE_ESTIMATORS = [GaussianNB, MLPClassifier]
PROTECTION_PERIODS = [50, 100, 200]
WINDOW_SIZE = [10, 20, 40]
UPDATE_PERIODS = [50, 100, 200]

metrics = (balanced_accuracy_score,)


def prepare_classifiers(ensemble_size):
    n_estimators = ensemble_size
    # online_bagging_clf = OnlineBagging(n_estimators=n_estimators, base_estimator=GaussianNB())
    # uob_clf = UOB(n_estimators=n_estimators, base_estimator=GaussianNB())
    # oob_clf = OOB(n_estimators=n_estimators, base_estimator=GaussianNB())
    # online_boosting_clf = OnlineBoosting(n_estimators=n_estimators, base_estimator=GaussianNB())
    ons_boosts = [
        ONSBoost(base_estimator=base_clf(),
                 n_estimators=n_estimators,
                 update_period=update_period,
                 window_size=window_size,
                 protection_period=protection_period)
        for protection_period in PROTECTION_PERIODS
        for window_size in WINDOW_SIZE
        for update_period in UPDATE_PERIODS
        for base_clf in BASE_ESTIMATORS
    ]
    all_clfs = [*ons_boosts]
    return n_estimators, all_clfs


def process_file(filename, dirpath, ensemble_size):
    prepared_classifiers = prepare_classifiers(ensemble_size)
    start = datetime.now()
    print(f"{start.strftime('%H:%M:%S')} starting with {filename}")
    n = prepared_classifiers[0]
    classifiers = prepared_classifiers[1]
    try:
        abspath = os.path.abspath(os.path.join(SAVED_STREAMS_PATH, dirpath, filename))
        evaluator = TestThenTrain(metrics=metrics, verbose=True)
        evaluator.process(NPYParser(abspath), classifiers)
        save_path = os.path.abspath(os.path.join(SAVED_SCORES_PATH, dirpath, str(n) + "_" + filename))
        np.save(arr=evaluator.scores, file=save_path)
    except Exception as e:
        print(f"{datetime.now().strftime('%H:%M:%S')} - {os.getpid()} - error during processing {n} and {filename} - {e}")
    print(f"{datetime.now().strftime('%H:%M:%S')} done with {filename} "
          f"(duration: {'{:.3f}'.format((datetime.now() - start).total_seconds())})")


def process_ddi_file_3(filename):
    return process_file(filename, DDI_PATH, 3)


def process_ddi_file_5(filename):
    return process_file(filename, DDI_PATH, 5)


def process_ddi_file_10(filename):
    return process_file(filename, DDI_PATH, 10)


if __name__ == '__main__':
    print(f"{datetime.now().strftime('%H:%M:%S')} starting")

    GENERATE_STREAMS = False
    GENERATE_RESULTS = False
    multiprocessing.set_start_method('spawn', force=True)

    if GENERATE_STREAMS:
        for random_state in RANDOM_STATES:
            for w in DDI_WEIGHTS:
                stream, filepath = prepare_di_stream(random_state, w)
                stream.save_to_npy(os.path.join(SAVED_STREAMS_PATH, DDI_PATH, filepath))
            for w in CDI_WEIGHTS:
                stream, filepath = prepare_di_stream(random_state, w)
                stream.save_to_npy(os.path.join(SAVED_STREAMS_PATH, CDI_PATH, filepath))
                stream, filepath = prepare_disco_stream(random_state, w)
                stream.save_to_npy(os.path.join(SAVED_STREAMS_PATH, DISCO_PATH, filepath))

    if GENERATE_RESULTS:
        freeze_support()
        with ProcessPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
            for filename in list(os.listdir(os.path.join(SAVED_STREAMS_PATH, DDI_PATH))):
                pool.submit(process_ddi_file_3, filename)
                pool.submit(process_ddi_file_5, filename)
                pool.submit(process_ddi_file_10, filename)

    for c in prepare_classifiers(3)[1]:
        print(c)
    #
    # ons_boosts = [
    #     ONSBoost(base_estimator=GaussianNB(),
    #              n_estimators=3,
    #              update_period=update_period,
    #              window_size=window_size,
    #              protection_period=protection_period)
    #     for protection_period in PROTECTION_PERIODS
    #     for window_size in WINDOW_SIZE
    #     for update_period in UPDATE_PERIODS
    # ]
    # for ob in ons_boosts:
    #     print(f"\"{ob}\"", end=', ')
