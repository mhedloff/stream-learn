import os

import numpy as np
from strlearn.streams import StreamGenerator

N_FEATURES = 10

RANDOM_STATES = [1000, 100000, 101010,
                 10110, 101101, 1001,
                 10101010, 101, 110, 1337]

DDI_WEIGHTS = [(.2, .9), (.2, .7)]
CDI_WEIGHTS = [(2, 5, .75), (2, 4, .9)]
DISCO_WEIGHTS = [(2, 5, 0.9)]
N_CHUNKS = 250
N_SAMPLES = 200
# N_CHUNKS = 1000
# N_SAMPLES = 100
DATA_STREAMS_PATH = './data_streams/'


def format_cdi(cdi):
    return f"w{str(cdi[0]).replace('.', ',')}_{str(cdi[1]).replace('.', ',')}_{str(cdi[2]).replace('.', ',')}"


def format_ddi(ddi):
    return f"w{str(ddi[0]).replace('.', ',')}_{str(ddi[1]).replace('.', ',')}"


if __name__ == '__main__':
    for rst in RANDOM_STATES:
        for cdi in CDI_WEIGHTS:
            stream = StreamGenerator(weights=cdi, random_state=rst, n_chunks=N_CHUNKS, chunk_size=N_SAMPLES,
                                     n_features=N_FEATURES)
            stream_name = f"cdi__{format_cdi(cdi)}__NC_{str(N_CHUNKS)}__CS_{str(N_SAMPLES)}__RST_{str(rst)}"
            stream.save_to_npy(os.path.join(DATA_STREAMS_PATH, 'cdi/', stream_name))

        for ddi in DDI_WEIGHTS:
            stream = StreamGenerator(weights=ddi, random_state=rst, n_chunks=N_CHUNKS, chunk_size=N_SAMPLES,
                                     n_features=N_FEATURES)
            stream_name = f"ddi__{format_ddi(ddi)}__NC_{str(N_CHUNKS)}__CS_{str(N_SAMPLES)}__RST_{str(rst)}"
            stream.save_to_npy(os.path.join(DATA_STREAMS_PATH, 'ddi/', stream_name))

        for disco in DISCO_WEIGHTS:
            stream = StreamGenerator(weights=disco, random_state=rst, n_chunks=N_CHUNKS, chunk_size=N_SAMPLES,
                                     recurring=True, incremental=True, n_drifts=3, concept_sigmoid_spacing=5,
                                     n_features=N_FEATURES)
            stream_name = f"disco__{format_cdi(disco)}__NC_{str(N_CHUNKS)}__CS_{str(N_SAMPLES)}__RST_{str(rst)}"
            stream.save_to_npy(os.path.join(DATA_STREAMS_PATH, 'disco/', stream_name))
