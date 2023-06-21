import os

import pandas as pd

disco_files = [f for f in os.listdir('./final/parameters_results/disco')
               if not os.path.isdir(f) and f.find('SGDClassifier') > 0]
cdi_files = [f for f in os.listdir('./final/parameters_results/cdi')
               if not os.path.isdir(f) and f.find('SGDClassifier') > 0]

RANDOM_STATES = [1000, 100000, 101010,
                 10110, 101101, 1001,
                 10101010, 101, 110, 1337]
STREAMS_NAMES = [
    'cdi__w2_4_0,9',
    'cdi__w2_5_0,75',
    'disco__w2_5_0,9'
]
f_s = { sn: { rst: [] for rst in RANDOM_STATES } for sn in STREAMS_NAMES}

for f in [*cdi_files, *disco_files]:
    sname = f.split('++')[1].split('__NC')[0]
    rst = int(f.split('__RST_')[1].split('.npy')[0])
    f_s[sname][rst].append(f)

c_s = { rst: { sn: len(f_s[sn][rst]) for sn in STREAMS_NAMES} for rst in RANDOM_STATES}
print(pd.DataFrame(c_s).to_string())

