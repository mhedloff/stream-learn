import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import freeze_support
from time import sleep
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
PROTECTION_PERIODS = [50, 100, 200]
WINDOW_SIZE = [10, 20, 40]
UPDATE_PERIODS = [50, 100, 200]
ENSEMBLE_SIZE = [5, 30, 100]
N_CHUNKS = 1000
N_SAMPLES = 100
STREAMS_LOCATION = './data_streams/'
RESULTS_LOCATION = './parameters_results/'


def ensemble_params_test(ensemble, stream_path):
    evaluator = TestThenTrain(metrics=(balanced_accuracy_score,), verbose=True)
    start_time = time.time()
    Logger.start(f"starting {stream_path} using {ensemble}")
    evaluator.process(NPYParser(stream_path, chunk_size=N_SAMPLES, n_chunks=N_CHUNKS), (ensemble,))
    Logger.end(f"end with {stream_path}, used {ensemble}, duration: {(time.time() - start_time):.3f}s")
    save_path = os.path.abspath(os.path.join(RESULTS_LOCATION, f'{ensemble}++{stream_path}'))


def f_test(s):
    start_time = time.time()
    Logger.start(f"starting with {s}")
    sleep(s)
    Logger.end(f"end with {s}, duration: {(time.time() - start_time):.3f}s")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    data_streams_dir = os.listdir(os.path.join(STREAMS_LOCATION))
    for base_estimator in BASE_ESTIMATORS:
        for protection_period in PROTECTION_PERIODS:
            for window_size in WINDOW_SIZE:
                for update_period in UPDATE_PERIODS:
                    for ensemble_size in ENSEMBLE_SIZE:
                        ons_boost = ONSBoost(base_estimator=base_estimator(), n_estimators=ensemble_size,
                                             update_period=update_period, protection_period=protection_period,
                                             window_size=window_size)
                        print(ons_boost)
    # for d_stream in [data_streams_dir[0]]:
    #     stream_abspath = os.path.abspath(os.path.join(STREAMS_LOCATION, d_stream))
    #     print(stream_abspath, not os.path.isdir(d_stream))
    #     estimator = ONSBoost(base_estimator=GaussianNB())
    #     ensemble_params_test(estimator, stream_abspath)

    #
    # with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as pool:
    #     tasks = [pool.submit(f_test, _) for _ in [1]]
    #     tasks_n = len(tasks)
    #     for _ in as_completed(tasks):
    #         pending_tasks_n = len(pool._pending_work_items)
    #         Logger.info(f'{pending_tasks_n} left. Progress: {tasks_n - pending_tasks_n}/{tasks_n}')
