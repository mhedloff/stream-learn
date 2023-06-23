import multiprocessing
import os
import signal
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

from strlearn.ensembles import OOB, UOB, OnlineBagging, OnlineBoosting, ONSBoost
from strlearn.evaluators import TestThenTrain
from strlearn.evaluators.TestThenTrain import Logger
from strlearn.metrics import balanced_accuracy_score, recall, precision, specificity, f1_score, geometric_mean_score_1
from strlearn.streams import NPYParser

METRICS = (recall, specificity, precision, balanced_accuracy_score, f1_score, geometric_mean_score_1)

RANDOM_STATES = [1000, 100000, 101010,
                 10110, 101101]
# RANDOM_STATES = [1000, 100000, 101010,
#                  10110, 101101, 1001,
#                  10101010, 101, 110, 1337]
N_CHUNKS = 125
N_SAMPLES = 200
STREAMS_LOCATION = os.path.join('./final/comparison_streams/')
RESULTS_LOCATION = os.path.join('./final/comparison_results/')


# def prepare_other_estimators(base):
#     return [OnlineBagging(base_estimator=base(), n_estimators=20),
#             OnlineBoosting(base_estimator=base(), n_estimators=20),
#             OOB(base_estimator=base(), n_estimators=20),
#             UOB(base_estimator=base(), n_estimators=20)]

def prepare_other_estimators_sgd(base):
    return [OnlineBagging(base_estimator=base(loss='log_loss'), n_estimators=20),
            OnlineBoosting(base_estimator=base(loss='log_loss'), n_estimators=20),
            OOB(base_estimator=base(loss='log_loss'), n_estimators=20),
            UOB(base_estimator=base(loss='log_loss'), n_estimators=20)]


cases = [
    {
        'stream_prefix': 'cdi__w2_4_0,9__NC_250__CS_200',
        'ensembles': {
#             'GaussianNB': lambda: [
#                 ONSBoost(base_estimator=GaussianNB(), n_estimators=10, protection_period=100, update_period=200,
#                          window_size=10),
#                 *prepare_other_estimators(GaussianNB)
#             ],
            'SGDClassifier': lambda: [
                ONSBoost(base_estimator=SGDClassifier(loss='log_loss'), n_estimators=30, protection_period=200,
                         update_period=100, window_size=10),
                *prepare_other_estimators_sgd(SGDClassifier)
            ]
        }
    },
    {
        'stream_prefix': 'cdi__w2_5_0,75__NC_250__CS_200',
        'ensembles': {
#             'GaussianNB': lambda: [
#                 ONSBoost(base_estimator=GaussianNB(), n_estimators=10, protection_period=50, update_period=200,
#                          window_size=10),
#                 *prepare_other_estimators(GaussianNB)
#             ],
            'SGDClassifier': lambda: [
                ONSBoost(base_estimator=SGDClassifier(loss='log_loss'), n_estimators=30, protection_period=100,
                         update_period=50, window_size=10),
                *prepare_other_estimators_sgd(SGDClassifier)
            ]
        }
    },
    {
        'stream_prefix': 'disco__w2_5_0,9__NC_250__CS_200',
        'ensembles': {
#             'GaussianNB': lambda: [
#                 ONSBoost(base_estimator=GaussianNB(), n_estimators=5, protection_period=50, update_period=200,
#                          window_size=10),
#                 *prepare_other_estimators(GaussianNB)
#             ],
            'SGDClassifier': lambda: [
                ONSBoost(base_estimator=SGDClassifier(loss='log_loss'), n_estimators=30, protection_period=50,
                         update_period=50, window_size=40),
                *prepare_other_estimators_sgd(SGDClassifier)
            ]
        }
    }
]


def ensemble_params_test(ensemble_supplier, stream_name, case_name):
    try:
        Logger.start(f"starting {stream_name}")
        # prepare evaluator
        evaluator = TestThenTrain(metrics=METRICS, verbose=True)
        stream_path = os.path.abspath(os.path.join(STREAMS_LOCATION, stream_name))
        stream = NPYParser(stream_path, chunk_size=N_SAMPLES, n_chunks=N_CHUNKS)

        # start processing
        start_time = time.time()
        evaluator.process(stream, clfs=ensemble_supplier)
        Logger.end(f" {case_name} :: end with {stream_name}, duration: {(time.time() - start_time):.3f}s")

        # saving scores
        save_path = os.path.abspath(os.path.join(RESULTS_LOCATION, f'v1{case_name}++{stream_name}'))
        Logger.start(f"saving {case_name} results under {save_path}")
        np.save(file=save_path, arr=evaluator.scores)
        Logger.end(f"saved {save_path}")
    except Exception as e:
        Logger.error(f"Error during processing {stream_name} with {case_name} - {e}")
    finally:
        Logger.end(f"finished processing {stream_name} with {case_name}")


def shutdown_pool(pool: ProcessPoolExecutor):
    Logger.error("CANCELLED ALL FUTURES")
    pool.shutdown(wait=True, cancel_futures=True)
    sys.exit(0)


def sigint_handler(arg: ProcessPoolExecutor):
    return lambda sig, frame: shutdown_pool(arg)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    freeze_support()
    multiprocessing.set_start_method('spawn', force=True)

    data_streams_dir = [
        f for f in os.listdir(os.path.abspath(STREAMS_LOCATION)) if not os.path.isdir(f)
    ]
    with ProcessPoolExecutor(max_workers=8, max_tasks_per_child=1) as pool:
    # with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        signal.signal(signal.SIGINT, sigint_handler(pool))
        tasks = []
        tasks_n = len(data_streams_dir) * len(cases) * 2
        for case in cases:
            for d_stream in data_streams_dir:
                if d_stream.startswith(case['stream_prefix']):
                    for k, v in case['ensembles'].items():
                        tasks.append(pool.submit(ensemble_params_test, v(), d_stream, k))

        for _ in as_completed(tasks):
            _.done()
            pending_tasks_n = len(pool._pending_work_items)
            Logger.info(f'{pending_tasks_n} left. Progress: {tasks_n - pending_tasks_n}/{tasks_n}')
