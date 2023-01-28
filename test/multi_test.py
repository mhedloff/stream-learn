import os.path
from multiprocessing.queues import Queue

import numpy as np
from matplotlib import pyplot as plt
from numpy import iterable
from sklearn.naive_bayes import GaussianNB

from strlearn.ensembles import ONSBoost, UOB, RandomClassifier, OnlineBoosting, OOB, OnlineBagging
from strlearn.evaluators import TestThenTrain
from strlearn.streams import StreamGenerator, NPYParser
from strlearn.metrics import balanced_accuracy_score, f1_score, recall

import warnings

GENERATE_STREAMS = False
SAVED_STREAMS_PATH = "D:\\JavaTraining\\mgr\\saved_streams"
SAVED_STREAMS_PATH = "D:\\JavaTraining\\mgr\\saved_scores"
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
    np.random.seed(seed)
    return StreamGenerator(weights=weight,
                           random_state=seed,
                           chunk_size=CHUNK_SIZE,
                           n_chunks=N_CHUNKS,
                           n_features=N_FEATURES), \
        f"w{weight}_s{seed}_nc{N_CHUNKS}_nf{N_FEATURES}_cs{CHUNK_SIZE}"


def prepare_disco_stream(seed, weight):
    np.random.seed(seed)
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


RANDOM_STATES = [42, 1337, 55_010]
N_ESTIMATORS = [3, 5, 10]
PROTECTION_PERIODS = [50, 100, 200]
WINDOW_SIZE = [10, 100, 200]
UPDATE_PERIODS = [50, 100, 200]

# queue = Queue()
processes = []
metrics = (balanced_accuracy_score,)

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


def prepare_classifiers():
    all_clfs = []
    for n_estimators in N_ESTIMATORS:
        online_bagging_clf = OnlineBagging(n_estimators=n_estimators, base_estimator=GaussianNB())
        uob_clf = UOB(n_estimators=n_estimators, base_estimator=GaussianNB())
        oob_clf = OOB(n_estimators=n_estimators, base_estimator=GaussianNB())
        online_boosting_clf = OnlineBoosting(n_estimators=n_estimators, base_estimator=GaussianNB())
        ons_boosts = [
            # ONSBoost(base_estimator=GaussianNB(),
            #          n_estimators=n_estimators,
            #          update_period=update_period,
            #          window_size=window_size,
            #          protection_period=protection_period)
            # for protection_period in PROTECTION_PERIODS
            # for window_size in WINDOW_SIZE
            # for update_period in UPDATE_PERIODS
        ]
        all_clfs.append([online_bagging_clf, uob_clf])  # , oob_clf, online_boosting_clf, *ons_boosts])
    return all_clfs


def process_file(filename, dirpath):
    all_clfs = prepare_classifiers()
    for clfs in all_clfs:
        abspath = os.path.abspath(os.path.join(SAVED_STREAMS_PATH, dirpath, filename))
        evaluator = TestThenTrain(metrics=metrics, verbose=True)
        evaluator.process(NPYParser(abspath), clfs)
        np.save(evaluator.scores[:, :, 0])


for path in os.listdir(os.path.join(SAVED_STREAMS_PATH, DDI_PATH)):
    process_file(path, DDI_PATH)
