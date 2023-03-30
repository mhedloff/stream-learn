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

SAVED_STREAMS_PATH = "D:\\JavaTraining\\mgr\\ensemble_size_str"
SAVED_SCORES_PATH = "D:\\JavaTraining\\mgr\\ensemble_size"
CDI_PATH = 'cdi'
DDI_PATH = 'ddi'
DISCO_PATH = 'disco'

N_DRIFTS = 3
CONCEPT_SIGMOID_SPACING = 5
N_CHUNKS = 150
N_FEATURES = 7
CHUNK_SIZE = 150

warnings.filterwarnings("ignore")

DDI_WEIGHTS = [
    [.2, .8],
    # [.3, .6],
    # [.1, .9],
    # [.2, .9]
]
# CDI_WEIGHTS = [
#     (2, 4, .8),
#     (2, 5, .9),
#     (3, 4, .8),
#     (3, 5, .9)
# ]


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


# RANDOM_STATES = [1000, 100000, 101010,
#                  10110, 101101, 1001,
#                  10101010, 101, 110, 1337]
RANDOM_STATES = [1000]
BASE_ESTIMATORS = [GaussianNB, MLPClassifier]
# PROTECTION_PERIODS = [50, 100, 200]
# WINDOW_SIZE = [10, 20, 40]
# UPDATE_PERIODS = [50, 100, 200]

metrics = (balanced_accuracy_score,)


def prepare_classifiers(base_clf):
    ons_boosts = [
        ONSBoost(base_estimator=base_clf(),
                 n_estimators=n_estimators,
                 update_period=200,
                 window_size=20,
                 protection_period=200)
        for n_estimators in [3, 5, 10, 20, 30, 40, 50]
    ]
    all_clfs = [*ons_boosts]
    return all_clfs


def process_file(filename, dirpath, base_clf):
    prepared_classifiers = prepare_classifiers(base_clf)
    start = datetime.now()
    print(f"{start.strftime('%H:%M:%S')} starting with {filename}")
    classifiers = prepared_classifiers
    try:
        abspath = os.path.abspath(os.path.join(SAVED_STREAMS_PATH, dirpath, filename))
        evaluator = TestThenTrain(metrics=metrics, verbose=True)
        evaluator.process(NPYParser(abspath), classifiers)
        save_path = os.path.abspath(os.path.join(SAVED_SCORES_PATH, dirpath, 'ESIZE_' + str(base_clf()) + filename))
        np.save(arr=evaluator.scores, file=save_path)
    except Exception as e:
        print(f"{datetime.now().strftime('%H:%M:%S')} - {os.getpid()} - error during processing {filename} - {e}")
    print(f"{datetime.now().strftime('%H:%M:%S')} done with {filename} "
          f"(duration: {'{:.3f}'.format((datetime.now() - start).total_seconds())})")


def process_ddi_file_mlp(filename):
    return process_file(filename, DDI_PATH, GaussianNB)


def process_ddi_file_gnb(filename):
    return process_file(filename, DDI_PATH, MLPClassifier)


if __name__ == '__main__':
    print(f"{datetime.now().strftime('%H:%M:%S')} starting")

    GENERATE_STREAMS = True
    GENERATE_RESULTS = True
    multiprocessing.set_start_method('spawn', force=True)

    if GENERATE_STREAMS:
        for random_state in RANDOM_STATES:
            for w in DDI_WEIGHTS:
                stream, filepath = prepare_di_stream(random_state, w)
                stream.save_to_npy(os.path.join(SAVED_STREAMS_PATH, DDI_PATH, filepath))

    if GENERATE_RESULTS:
        process_ddi_file_gnb('w[0.2, 0.8]_s1000_nc150_nf7_cs150.npy')
        # freeze_support()
        # with ProcessPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
        #     for filename in list(os.listdir(os.path.join(SAVED_STREAMS_PATH, DDI_PATH))):
        #         pool.submit(process_ddi_file_gnb, filename)

    for c in prepare_classifiers():
        print(c)
