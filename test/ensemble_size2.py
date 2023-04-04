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

ENSEMBLE_SIZE = [3, 5, 10, 20, 30, 40, 50, 75, 100]

SAVED_STREAMS_PATH = "D:\\projects\\mgr\\ensemble_size_str"
SAVED_SCORES_PATH = "D:\\projects\\mgr\\ensemble_size"
CDI_PATH = 'cdi'
DDI_PATH = 'ddi'
DISCO_PATH = 'disco'

N_DRIFTS = 3
CONCEPT_SIGMOID_SPACING = 5
N_CHUNKS = 100
N_FEATURES = 7
CHUNK_SIZE = 150

warnings.filterwarnings("ignore")

DDI_WEIGHTS = [
    [.2, .8],
]


def prepare_di_stream(seed, weight):
    return StreamGenerator(weights=weight,
                           random_state=seed,
                           chunk_size=CHUNK_SIZE,
                           n_chunks=N_CHUNKS,
                           n_features=N_FEATURES), \
        f"w{weight}_s{seed}_nc{N_CHUNKS}_nf{N_FEATURES}_cs{CHUNK_SIZE}"


RANDOM_STATES = [1000, 100000, 101010,
                 10110, 101101, 1001,
                 10101010, 101, 110, 1337]

BASE_ESTIMATORS = [GaussianNB, MLPClassifier]

metrics = (balanced_accuracy_score,)


def prepare_classifiers(base_clf, _n):
    ons_boosts = [
        ONSBoost(base_estimator=base_clf(),
                 n_estimators=_n,
                 update_period=200,
                 window_size=20,
                 protection_period=200)
    ]
    all_clfs = [*ons_boosts]
    return all_clfs


def process_stream(base_clf, _n, filename, parsed_stream):
    start = datetime.now()
    print(f"{os.getpid()}:: {start.strftime('%H:%M:%S')} starting with {filename}")
    try:
        evaluator = TestThenTrain(metrics=metrics, verbose=True)
        evaluator.process(parsed_stream, prepare_classifiers(base_clf, _n))
        save_path = os.path.abspath(os.path.join(SAVED_SCORES_PATH, DDI_PATH, filename))
        np.save(arr=evaluator.scores, file=save_path)
    except KeyError as e:
        print(f"{os.getpid()}:: {datetime.now().strftime('%H:%M:%S')} "
              f"- {os.getpid()} - error during processing {filename} - {e}")
    print(f"{os.getpid()}:: {datetime.now().strftime('%H:%M:%S')} done with {filename} "
          f"(duration: {'{:.3f}'.format((datetime.now() - start).total_seconds())})")


def f(_base_clf, _n, _rs):
    generated_stream = prepare_di_stream(_rs, weight=DDI_WEIGHTS[0])[0]
    return process_stream(_base_clf, _n, f'{str(_base_clf())}_{_n}_{_rs}', generated_stream)


if __name__ == '__main__':
    print(f"{datetime.now().strftime('%H:%M:%S')} starting")

    GENERATE_RESULTS = True
    multiprocessing.set_start_method('spawn', force=True)

    if GENERATE_RESULTS:
        freeze_support()
        # with ProcessPoolExecutor(1) as pool:
        with ProcessPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
            for _n in ENSEMBLE_SIZE:
                for _rs in RANDOM_STATES:
                    for clf in [GaussianNB, MLPClassifier]:
                        # (f(clf, _n, _rs))
                        pool.submit(f, clf, _n, _rs)

            # pool.submit(process_gnb_stream, prepare_di_stream(1000, [.2, .9])[0])
            # pool.submit(process_mlp_stream, prepare_di_stream(1000, [.2, .9])[0])
            # for filename in list(os.listdir(os.path.join(SAVED_STREAMS_PATH, DDI_PATH))):
            #     pass
    #
    # for c in prepare_classifiers():
    #     print(c)
