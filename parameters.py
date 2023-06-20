import functools
import multiprocessing
import operator
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
from sklearn.neural_network import MLPClassifier
from strlearn.ensembles import ONSBoost
from strlearn.evaluators import TestThenTrain
from strlearn.evaluators.TestThenTrain import Logger
from strlearn.metrics import balanced_accuracy_score
from strlearn.streams import NPYParser

RANDOM_STATES = [1337]

# RANDOM_STATES = [1000, 100000, 101010,
#                  10110, 101101, 1001,
#                  10101010, 101, 110, 1337]
BASE_ESTIMATORS = [SGDClassifier]
METRICS = (balanced_accuracy_score,)
PROTECTION_PERIODS = [50, 100, 200]
WINDOW_SIZE = [10, 20, 40]
UPDATE_PERIODS = [50, 100, 200]
ENSEMBLE_SIZE = [5, 10, 30]
N_CHUNKS = 250
N_SAMPLES = 200
DIRECTORY = 'cdi/'
STREAMS_LOCATION = os.path.join('./final/data_streams/', DIRECTORY)
RESULTS_LOCATION = os.path.join('./final/parameters_results/', DIRECTORY)


def ensemble_params_test(ensemble, stream_name):
    try:
        save_path = os.path.abspath(os.path.join(RESULTS_LOCATION, f'{ensemble}++{stream_name}'))
        if os.path.exists(save_path):
            Logger.end(f'no processing. File <<{save_path}>> already exists.')
            return
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
    data_streams_dir = [
        f for f in data_streams_dir if int(f.split('.')[0].split('__RST_')[1]) in RANDOM_STATES
    ]
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1, max_tasks_per_child=1) as pool:
        signal.signal(signal.SIGINT, sigint_handler(pool))
        tasks = []
        all_params = [data_streams_dir, BASE_ESTIMATORS, PROTECTION_PERIODS, WINDOW_SIZE, UPDATE_PERIODS, ENSEMBLE_SIZE]
        lengths = [len(x) for x in all_params]
        tasks_n = functools.reduce(operator.mul, lengths)
        done_tasks = 0
        for d_stream in data_streams_dir:
            for base_estimator in BASE_ESTIMATORS:
                for protection_period in PROTECTION_PERIODS:
                    for window_size in WINDOW_SIZE:
                        for update_period in UPDATE_PERIODS:
                            for ensemble_size in ENSEMBLE_SIZE:
                                estimator = ONSBoost(base_estimator=SGDClassifier(loss='log_loss'),
                                                     n_estimators=ensemble_size, update_period=update_period,
                                                     protection_period=protection_period, window_size=window_size)
                                if os.path.exists(os.path.join(RESULTS_LOCATION, f'{estimator}++{d_stream}')):
                                    done_tasks += 1
                                    Logger.info(f'skipped. Progress: {done_tasks}/{tasks_n}.\n'
                                                f'<<{estimator}++{d_stream}>> exists')
                                else:
                                    tasks.append(pool.submit(ensemble_params_test, estimator, d_stream))

        for _ in as_completed(tasks):
            _.done()
            pending_tasks_n = len(pool._pending_work_items)
            done_tasks += 1
            Logger.info(f'{pending_tasks_n} left. Progress: {done_tasks}/{tasks_n}')
