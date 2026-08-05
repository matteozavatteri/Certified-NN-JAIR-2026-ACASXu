import os 
import json

PROPERTIES = [1,2,3,4,5,6,7,8,9,10]

BRAINS = [(6,50)]
LEARNING_RATES = [1e-3]
SEEDS = [100]

STRATEGIES = [(50,20)]

SOFT = ['-SA']
EPSILON = [1e-4]
DELTA   = [1e-3]

PRA = [1,2,3,4,5]
# time to loss of vertical separation
TAU = [1,2,3,4,5,6,7,8,9]

key_template = (
    '{pra}-{tau}-'
    '{property}-'
    '{epsilon}-'
    '{delta}-'
    '{learning_rate}-'
    '{layers}x{neurons}-'
    '{counterexamples}-'
    '{sampling}-'
    '{seed}-L'
    '{soft}'
    )


KEYS = [key_template.format(
    pra=pra,
    tau=tau,
    property=p, 
    epsilon=epsilon,
    delta=delta, 
    learning_rate=learning_rate, 
    layers=layers,
    neurons=neurons,
    counterexamples=counterexamples, 
    sampling=sampling, 
    seed=seed, 
    soft=soft
    ) 
    for p in PROPERTIES 
    for pra in PRA 
    for tau in TAU 
    for epsilon in EPSILON 
    for delta in DELTA 
    for learning_rate in LEARNING_RATES 
    for (layers,neurons) in BRAINS 
    for (counterexamples,sampling) in STRATEGIES 
    for seed in SEEDS 
    for soft in SOFT 
    if delta > epsilon and (
        (p in [1,2,3,4,5,6,10]) or 
        (p == 7 and tau == 9) or 
        (p == 8 and pra == 2 and tau == 9) or 
        (p == 9 and pra == 3))]


def load_experiment_data():
    data = dict()
    for key in KEYS:
        path = 'experiments/{}'.format(key)
        if os.path.exists(path):
            with open('{}/stats.json'.format(path), "r") as f:
                data[key] = json.load(f)
    return data

if __name__ == '__main__':
    data = load_experiment_data()
    total = len(data)
    missing = [k for k in KEYS if k not in data]
    timeout = [k for k in data.keys() if data[k]['TotalTIME'] == None]

    print('total={}'.format(total))
    print('missing={}'.format(missing))
    print('timeout={}'.format(timeout))