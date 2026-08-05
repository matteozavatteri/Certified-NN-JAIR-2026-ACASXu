import subprocess

# in seconds
TIMEOUT=3600*5

P = [1,2,3,4,5]
E = 1000 # epochs
S = 50  # early stopping

EPSILON = [1e-4]
DELTA = [1e-3]

# layers x neurons
H = [(6,50)]

# learning rate
LR = [1e-3]

# seeds
SEEDS = [100]

# strategy (counterexamples, sampling)
STRATEGY = [(50,20)]

# previous action
PRA = ['COC', 'WL', 'WR', 'SL', 'SR']

# time to loss of vertical separation
TAU = ['0','1','5','10','20','40','60','80','100']

SOFT = ['--soft-acceleration']

# command
c = (
    'python ACASXuCertifier.py {property} '
    '--pra {pra} '
    '--tau {tau} '
    '--learning-rate {lr} '
    '--epochs {epochs} '
    '--early-stopping {earlystopping} '
    '--hidden-layers {layers} '
    '--neurons-per-hidden-layer {neurons} '
    '--counterexamples {counterexamples} '
    '--sampling {sampling} '
    '--seed {seed} '
    '{soft} ' 
    '--epsilon {epsilon} '
    '--delta {delta}'
    )

cmds = [c.format(
    property=p, 
    pra=pra, 
    tau=tau, 
    lr=lr, 
    epochs=E, 
    earlystopping=S, 
    layers=layers, 
    neurons=neurons,
    counterexamples=ce,
    sampling=sam, 
    seed=seed, 
    soft=soft, 
    epsilon=epsilon, 
    delta=delta) 
        for p in P
        for pra in PRA 
        for tau in TAU
        for lr in LR
        for (layers,neurons) in H 
        for seed in SEEDS
        for (ce,sam) in STRATEGY
        for soft in SOFT
        for epsilon in EPSILON
        for delta in DELTA
        if delta > epsilon and (
            (p in [1,2,3,4,5,6,10]) or
            (p == 7 and tau == '100') or
            (p == 8 and pra == 'WL' and tau == '100') or 
            (p == 9 and pra == 'WR') 
        )]

for cmd in cmds:
    #print("Running: {}".format(cmd))

    try:
        p = subprocess.run(cmd.split(), stdout=subprocess.PIPE, timeout=TIMEOUT)
    except subprocess.CalledProcessError as e:
        print("{}".format(e))
    except:
        print('timeout\n')
