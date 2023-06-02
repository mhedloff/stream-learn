import functools
import multiprocessing
import operator
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import freeze_support
from time import sleep

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from strlearn.ensembles import ONSBoost
from strlearn.evaluators import TestThenTrain
from strlearn.evaluators.TestThenTrain import Logger
from strlearn.metrics import balanced_accuracy_score
from strlearn.streams import NPYParser, StreamGenerator


RANDOM_STATES = [1000, 100000, 101010,
                 10110, 101101, 1001,
                 10101010, 101, 110, 1337]
BASE_ESTIMATORS = [GaussianNB, MLPClassifier]
METRICS = (balanced_accuracy_score,)
PROTECTION_PERIODS = [50, 100, 200]
WINDOW_SIZE = [10, 20, 40]
UPDATE_PERIODS = [50, 100, 200]
ENSEMBLE_SIZE = [5, 30, 100]
N_CHUNKS = 1000
N_SAMPLES = 100
STREAMS_LOCATION = './data_streams/'
RESULTS_LOCATION = './parameters_results/'


def ensemble_params_test(ensemble, stream_name):
    try:
        # prepare evaluator
        evaluator = TestThenTrain(metrics=METRICS, verbose=True)
        stream_path = os.path.abspath(os.path.join(STREAMS_LOCATION, stream_name))
        stream = NPYParser(stream_path, chunk_size=N_SAMPLES, n_chunks=N_CHUNKS)

        # start processing
        Logger.start(f"starting {stream_name} using {ensemble}")
        start_time = time.time()
        evaluator.process(stream, (ensemble,))
        Logger.end(f"end with {stream_name}, used {ensemble}, duration: {(time.time() - start_time):.3f}s")

        # saving scores
        save_path = os.path.abspath(os.path.join(RESULTS_LOCATION, f'{ensemble}++{stream_name}'))
        Logger.start(f"saving {ensemble} results under {save_path}")
        np.save(file=save_path, arr=evaluator.scores)
        Logger.end(f"saved {save_path}")
    except Exception as e:
        Logger.error(f"Error during processing {stream_name} with {ensemble} - {e}")
    finally:
        Logger.end(f"finished processing {stream_name} with {ensemble}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    freeze_support()
    multiprocessing.set_start_method('spawn', force=True)

    data_streams_dir = [f for f in os.listdir(os.path.join(STREAMS_LOCATION)) if not os.path.isdir(f)]
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as pool:
        tasks = []
        all_params = [data_streams_dir, BASE_ESTIMATORS, PROTECTION_PERIODS, WINDOW_SIZE, UPDATE_PERIODS, ENSEMBLE_SIZE]
        lengths = [len(x) for x in all_params]
        tasks_n = functools.reduce(operator.mul, lengths)
        for d_stream in data_streams_dir:
            for base_estimator in BASE_ESTIMATORS:
                for protection_period in PROTECTION_PERIODS:
                    for window_size in WINDOW_SIZE:
                        for update_period in UPDATE_PERIODS:
                            for ensemble_size in ENSEMBLE_SIZE:
                                estimator = ONSBoost(base_estimator=base_estimator(), n_estimators=ensemble_size,
                                                     update_period=update_period, protection_period=protection_period,
                                                     window_size=window_size)
                                tasks.append(pool.submit(ensemble_params_test, estimator, d_stream))
        for _ in as_completed(tasks):
            pending_tasks_n = len(pool._pending_work_items)
            Logger.info(f'{pending_tasks_n} left. Progress: {tasks_n - pending_tasks_n}/{tasks_n}')
