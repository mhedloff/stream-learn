from sklearn.naive_bayes import GaussianNB

from strlearn.ensembles import OnlineBoosting, ONSBoost
from strlearn.evaluators import TestThenTrainDiversity
from strlearn.streams import StreamGenerator

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
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from strlearn.ensembles import ONSBoost
from strlearn.evaluators import TestThenTrain
from strlearn.evaluators.TestThenTrain import Logger
from strlearn.metrics import balanced_accuracy_score
from strlearn.streams import NPYParser


RANDOM_STATES = [1000, 100000, 101010,
                 10110, 101101, 1001,
                 10101010, 101, 110, 1337]
BASE_ESTIMATORS = [GaussianNB, MLPClassifier]
ensemble_sizes = [5, 10, 20, 50, 75, 100]
N_CHUNKS = 100
CHUNK_SIZE = 100
DIRECTORY = 'cdi/'
STREAMS_LOCATION = os.path.join('./final/data_streams_diversity/', DIRECTORY)
RESULTS_LOCATION = os.path.join('./final/diversity_results/', DIRECTORY)


def ensemble_params_test(ensemble, stream_name):
    try:
        save_path = os.path.abspath(os.path.join(RESULTS_LOCATION, f'{ensemble}++{stream_name}'))
        if os.path.exists(save_path):
            Logger.end(f'no processing. File <<{save_path}>> already exists.')
            return

        # prepare evaluator
        evaluator = TestThenTrainDiversity(verbose=True)
        stream_path = os.path.abspath(os.path.join(STREAMS_LOCATION, stream_name))
        stream = NPYParser(stream_path, chunk_size=CHUNK_SIZE, n_chunks=N_CHUNKS)

        # start processing
        Logger.start(f"starting {stream_name} using {ensemble}")
        start_time = time.time()
        evaluator.process(stream, (ensemble,))
        Logger.end(f"end with {stream_name}, used {ensemble}, duration: {(time.time() - start_time):.3f}s")

        # saving scores
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

    data_streams_dir = [f for f in os.listdir(os.path.abspath(STREAMS_LOCATION)) if not os.path.isdir(f)]
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1, max_tasks_per_child=1) as pool:
        signal.signal(signal.SIGINT, sigint_handler(pool))
        tasks = []
        all_params = [data_streams_dir, BASE_ESTIMATORS, ensemble_sizes]
        lengths = [len(x) for x in all_params]
        tasks_n = functools.reduce(operator.mul, lengths)
        for d_stream in data_streams_dir:
            for base_estimator in BASE_ESTIMATORS:
                for ensemble_size in ensemble_sizes:
                    window_size = 10
                    protection_period = 200
                    update_period = 200
                    estimator = ONSBoost(base_estimator=base_estimator(), n_estimators=ensemble_size,
                                         update_period=update_period, protection_period=protection_period,
                                         window_size=window_size)
                    tasks.append(pool.submit(ensemble_params_test, estimator, d_stream))

        for _ in as_completed(tasks):
            _.done()
            pending_tasks_n = len(pool._pending_work_items)
            Logger.info(f'{pending_tasks_n} left. Progress: {tasks_n - pending_tasks_n}/{tasks_n}')

