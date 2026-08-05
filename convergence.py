import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import argparse
import math
import conf

import properties.P1 as P1
import properties.P2 as P2
import properties.P3 as P3
import properties.P4 as P4
import properties.P5 as P5
import properties.P6 as P6
import properties.P10 as P10
import properties.P12345610 as P12345610

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

#plt.rcParams['text.usetex'] = True

def neural_network(hidden_layers, neurons_per_hidden_layer):

	InputNeurons  = conf.InputSIZE
	OutputNeurons = conf.OutputSIZE

	layers = []
	layers.append(nn.Linear(InputNeurons,neurons_per_hidden_layer))
	layers.append(nn.ReLU())
	
	for _ in range(hidden_layers-1):
		layers.append(nn.Linear(neurons_per_hidden_layer,neurons_per_hidden_layer))
		layers.append(nn.ReLU())

	layers.append(nn.Linear(neurons_per_hidden_layer,OutputNeurons))
	network = nn.Sequential(*layers)
	
	return network


import os 
import json
import argparse
from statistics import mean, stdev


def CounterExamples(P,network,EPSILON,DELTA,N=100000):
	
	CE = 0

	if P == 1:	
		x = P1.precondition_enabling_batch(N)
		y = network(x)
		mask = (y[:, conf.coc] + EPSILON <= -0.0433 + DELTA)
		#mask = y[:, index_coc] + EPSILON <= -0.0433 + DELTA
		CE = mask.sum().item()

	elif P == 2:
		x = P2.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.coc] <= y[:, conf.ActionsDifferentFromCOC].min(dim=1).values + (DELTA - EPSILON)
		# tmp = EPSILON + y[:, index_coc] - DELTA
		# mask = (tmp <= y[:, index_wl]) & (tmp <= y[:, index_wr]) & (tmp <= y[:, index_sl]) & (tmp <= y[:, index_sr])
		CE = mask.sum().item()

	elif P == 3:
		x = P3.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.ActionsDifferentFromCOC].max(dim=1).values <= y[:, conf.coc] + (DELTA - EPSILON)
		# tmp = EPSILON - y[:, index_coc] - DELTA
		# mask = (tmp <= -y[:, index_wl]) & (tmp <= -y[:, index_wr]) & (tmp <= -y[:, index_sl]) & (tmp <= -y[:, index_sr])
		CE = mask.sum().item()

		
	elif P == 4:
		x = P4.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.ActionsDifferentFromCOC].max(dim=1).values <= y[:, conf.coc] + (DELTA - EPSILON)
		# tmp = EPSILON - y[:, index_coc] - DELTA
		# mask = (tmp <= -y[:, index_wl]) & (tmp <= -y[:, index_wr]) & (tmp <= -y[:, index_sl]) & (tmp <= -y[:, index_sr])
		CE = mask.sum().item()

	elif P == 5:
		x = P5.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.sr] <= y[:, conf.ActionsDifferentFromSR].max(dim=1).values + (DELTA - EPSILON)
		# tmp = EPSILON + y[:, index_sr] - DELTA
		# mask = (tmp <= y[:, index_coc]) | (tmp <= y[:, index_wl]) | (tmp <= y[:, index_wr]) | (tmp <= y[:, index_sl])
		CE = mask.sum().item()

	elif P == 6:
		x = P6.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.coc] <= y[:, conf.ActionsDifferentFromCOC].max(dim=1).values + (DELTA - EPSILON)
		# tmp = EPSILON + y[:, index_coc] - DELTA
		# mask = (tmp <= y[:, index_wl]) | (tmp <= y[:, index_wr]) | (tmp <= y[:, index_sl]) | (tmp <= y[:, index_sr])
		CE = mask.sum().item()

		# low[index_theta]  = 0.1114
		# high[index_theta] = 0.4999
		# x = low + (high - low) * torch.rand(int(N/2), low.shape[0])
		# y = network(x)
		# tmp = EPSILON + y[:, index_coc] - DELTA
		# mask = (tmp <= y[:, index_wl]) | (tmp <= y[:, index_wr]) | (tmp <= y[:, index_sl]) | (tmp <= y[:, index_sr])
		# CE += mask.sum().item()

	elif P == 10:
		x = P10.precondition_enabling_batch(N)
		y = network(x)
		mask = y[:, conf.coc] <= y[:, conf.ActionsDifferentFromCOC].max(dim=1).values + (DELTA - EPSILON)
		# tmp = EPSILON + y[:, index_coc] - DELTA
		# mask = (tmp <= y[:, index_wl]) | (tmp <= y[:, index_wr]) | (tmp <= y[:, index_sl]) | (tmp <= y[:, index_sr])
		CE = mask.sum().item()

		
	elif P == 12345610:
		ALL = [1,2,3,4,5,6,10]
		for P in ALL:
			CE += CounterExamples(P,network,EPSILON=EPSILON,DELTA=EPSILON,N=int(N/len(ALL)))

	return CE

if not os.path.exists("img"):
	os.mkdir("img")

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("property", type=int, help="1,2,3,4,5,6,10,12345610")
	parser.add_argument("--ablation", default=False, action='store_true', help="compute graphs for ablation")
	
	args = parser.parse_args()

	SEEDS = [100,200,300]
	DELTAS = [0.01,0.001,0.0001]   if args.ablation else [0.01]
	EPSILONS = [0.01,0.001,0.0001] if args.ablation else [0.0001]
	PRA = [0]
	TAU = ['00','05','10','15','20','30','40','60']

	COUNTEREXAMPLES = [50]
	#print('COUNTEREXAMPLES={}'.format(COUNTEREXAMPLES))
	SAMPLING = [20]
	#print('SAMPLING={}'.format(SAMPLING))

	SOFT = [''] if args.ablation else ['','-SA']

	DELTA_EPSILON = [(d,e) for d in DELTAS for e in EPSILONS if d > e]
	
	Networks = ['6x50']
	Rates = [0.001]


	DATA = dict()
	timeout = set()
	missing = set()
	xticks = 11
	print("Loading data...")

	print()
	print("Property P{}".format(args.property))
	for N in Networks:
		print("Network {}".format(N))
		LAYERS  = int(N.split('x')[0])
		NEURONS = int(N.split('x')[-1])
		for R in Rates:
			for pra in PRA:
				for tau in TAU:
					for SEED in SEEDS:
						for SA in SOFT:
							for (DELTA,EPSILON) in DELTA_EPSILON:
								for CEX in COUNTEREXAMPLES:
									for SAM in SAMPLING:
										key = 'pra{pra}-tau{tau}/{P}-{EPS}-{DEL}-{R}-{N}-{CEX}-{SAM}-{SEED}-L{SOFT}'.format(pra=pra,tau=tau,P=args.property,EPS=EPSILON,DEL=DELTA,R=R,N=N,CEX=CEX,SAM=SAM,SEED=SEED,SOFT=SA)
										path = 'experiments/{}'.format(key)
										if os.path.exists(path):
											with open('{}/stats.json'.format(path), "r") as f:
												STATS = json.load(f)
												CEGIS_iterations = STATS['CEGISiterations']
												if STATS['TotalTIME'] is not None:
													DATA[key] = []
													INDEXES = [round(i * (CEGIS_iterations - 1) / (xticks - 1)) for i in range(xticks)]
													for index in INDEXES:
														network = neural_network(LAYERS,NEURONS)
														network.load_state_dict(torch.load('{}/network-{}.pt'.format(path,index+1),map_location=torch.device(device)))
														with torch.inference_mode():
															CE = CounterExamples(args.property,network,EPSILON,DELTA)
															DATA[key].append(CE)
												else:
													timeout.add(key)
										else:
											missing.add(key)
	print("done")
	print('missing={}'.format(missing))
	print('timeout={}'.format(timeout))

	x = [10*i for i in range(xticks)]

	COLORS  = {
			(0.01,0.001)  : 'tab:blue',
			(0.01,0.0001)   : 'tab:red',
			(0.001,0.0001)   : 'tab:green',
			'10x6'  : 'tab:orange',
			'11x10' : 'tab:purple',
			}

	MARKERS  = {
			(0.01,0.001)   : 'o',
			(0.01,0.0001)  : 's',
			(0.001,0.0001) : 'X',
			'10x6'  : 'P',
			'11x10' : 'D',
			}

	print("Computing plots...")

	print('Property P{}'.format(args.property))
	fig, ax = plt.subplots()
	plt.xlabel('CEGIS iterations')
	plt.ylabel('Counterexamples')
	labels = {'xticks' : x, 'xticklabels': x}
	ax.set(**labels)
	for N in Networks:
		LAYERS  = int(N.split('x')[0])
		NEURONS = int(N.split('x')[-1])
		for (DELTA,EPSILON) in DELTA_EPSILON: 
			for SA in SOFT:                                                                                                                                                          
				KEYS = ['pra{pra}-tau{tau}/{P}-{EPS}-{DEL}-{R}-{N}-{CEX}-{SAM}-{SEED}-L{SOFT}'.format(pra=pra,tau=tau,P=args.property,EPS=EPSILON,DEL=DELTA,R=R,N=N,CEX=CEX,SAM=SAM,SEED=SEED,SOFT=SA) for pra in PRA for tau in TAU for R in Rates for SEED in SEEDS for CEX in COUNTEREXAMPLES for SAM in SAMPLING if 'pra{pra}-tau{tau}/{P}-{EPS}-{DEL}-{R}-{N}-{CEX}-{SAM}-{SEED}-L{SOFT}'.format(pra=pra,tau=tau,P=args.property,EPS=EPSILON,DEL=DELTA,R=R,N=N,CEX=CEX,SAM=SAM,SEED=SEED,SOFT=SA) in DATA.keys()]
				y = []
				for i in range(xticks):
					y.append(mean(DATA[k][i] for k in KEYS))

				epsilonTEXT = '10^{-2}' if EPSILON == 0.01 else '10^{-3}' if EPSILON == 0.001 else '10^{-4}'
				deltaTEXT   = '10^{-2}' if DELTA   == 0.01 else '10^{-3}' if DELTA   == 0.001 else '10^{-4}'
				label = '$\\Delta={}, \\varepsilon={}$'.format(deltaTEXT,epsilonTEXT) if SA == '' else None
				linestyle = 'solid' if SA == '' else 'dashed'
				ax.plot(x,y,marker=MARKERS[(DELTA,EPSILON)],color=COLORS[(DELTA,EPSILON)],linewidth=2,linestyle=linestyle,label=label)
		
	ax.xaxis.set_major_formatter(PercentFormatter(100))
	plt.legend()
	plt.grid(True)

	name = 'ablation' if args.ablation else 'baseline'
	print('Saving img/convergence-ACAS-{}-P{}.pdf'.format(name,args.property))
	plt.savefig('img/convergence-ACAS-{}-P{}.pdf'.format(name,args.property))

	print("done")
