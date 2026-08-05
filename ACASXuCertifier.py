import torch
import torch.nn as nn

import argparse
from statistics import mean

import conf
import os
import time
import shutil
import json
import sys
import random
import pandas as pd
import conf
import training
import fm

#device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("Using device: {}".format(device))
torch.set_default_device(device)

def load_dataset(dataset):
	f = h5py.File(dataset, 'r')
	return [X.tolist() + y.tolist() for (X,y) in zip(f['X'],f['y'])]

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

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("property", type=int, default=1, help="1,2,3,4,5,6,7,8,9,10")
	parser.add_argument("--learning-rate", type=float, default=0.001, help="learning rate (0.01, 0.001, 0.0001; default: 0.001)")
	parser.add_argument("--epochs", type=int, default=100, help="number of epochs (default: 100)")
	parser.add_argument("--early-stopping", type=int, default=100, help="early stopping (default: 100)")
	parser.add_argument("--hidden-layers", type=int, default=7, help="number of hidden layers (default: 7)")
	parser.add_argument("--neurons-per-hidden-layer", type=int, default=8, help="number of neurons per hidden layer (default: 8)")
	parser.add_argument("--seed", type=int, default=1, help="random seed for shuffling dataset (default: 1)")
	parser.add_argument("--counterexamples", type=int, default=1, help="max number of counterexamples")
	parser.add_argument("--sampling", type=int, default=20, help="sample more points around the repaired counterexample")
	parser.add_argument("--retain", default=False, action='store_true', help="retain the current network (last CEGIS iteration)")
	parser.add_argument("--force", default=False, action='store_true', help="delete previous stat files and do the experiment again")
	parser.add_argument("--soft-acceleration", default=False, action='store_true', help="apply soft constraint acceleration")
	parser.add_argument("--pra", default='COC', help="previous advisory: COC, WL, WR, SL, SR; default: 0)")
	parser.add_argument("--tau", default='0', help="time to loss of vertical separation (0,1,5,10,20,40,60,80,100: default: 0)")
	parser.add_argument("--keep", default=False, action='store_true', help="always keep the current network (last CEGIS iteration)")
	parser.add_argument("--delta", type=float, default=0.01, help="learning rate (0.01, 0.001, 0.0001; default: 0.01)")
	parser.add_argument("--epsilon", type=float, default=0.0001, help="learning rate (0.01, 0.001, 0.0001; default: 0.0001)")
	parser.add_argument("--cores", type=int, default=4, help="number of physical CPU cores to use where appropriate (e.g., formal verification)")
	args = parser.parse_args()

	assert args.property in [1,2,3,4,5,6,7,8,9,10]
	assert args.learning_rate >= 0.0001
	assert args.epochs >= 1
	assert args.early_stopping >= 1 and args.early_stopping <= args.epochs
	assert args.hidden_layers >= 1
	assert args.neurons_per_hidden_layer >= 1
	assert args.counterexamples > 0
	assert args.sampling >= 0
	assert args.delta >= 0
	assert args.epsilon >= 0
	assert args.cores >= 1

	PRA = ['COC', 'WL', 'WR', 'SL', 'SR']
	assert args.pra in PRA
	args.pra = PRA.index(args.pra) + 1

	TAU = ['0','1','5','10','20','40','60','80','100']
	assert args.tau in TAU
	args.tau = TAU.index(args.tau) + 1

	#print(args.pra)
	#print(args.tau)

	if not os.path.exists('experiments'):
		os.mkdir('experiments')

	EXPERIMENT = '{pra}-{tau}-{property}-{epsilon}-{delta}-{lr}-{layers}x{neurons}-{counterexamples}-{sampling}-{seed}-{keep}{soft}'.format(
		pra=args.pra,
		tau=args.tau,
		property=args.property,
		epsilon=args.epsilon,
		delta=args.delta,
		lr=args.learning_rate,
		layers=args.hidden_layers,
		neurons=args.neurons_per_hidden_layer,
		counterexamples=args.counterexamples,
		sampling=args.sampling,
		seed=args.seed,
		keep='K' if args.keep else 'L',
		soft='-SA' if args.soft_acceleration else ''
		)
	
	if os.path.exists('experiments/{}'.format(EXPERIMENT)):
		if not args.force:
			exit('experiment exists, use --force to overwrite and do the experiment again')
		else:
			shutil.rmtree('experiments/{}'.format(EXPERIMENT))
	os.mkdir('experiments/{}'.format(EXPERIMENT))
	
	START = time.time()

	dataset = 'datasets/ACASXU_run2a_{pra}_{tau}_batch_2000.csv'.format(pra=args.pra,tau=args.tau)
	df = pd.read_csv(dataset)
	df = df.sample(frac=1, random_state=args.seed)
		
	# partition
	Training = df.iloc[:8000]
	Validation = df.iloc[8000:9000]
	Test = df.iloc[9000:]

	assert len(df) == (len(Training) + len(Validation) + len(Test))

	X_train = torch.tensor(Training.iloc[:,:conf.InputSIZE].values, dtype=torch.float32)
	y_train = torch.tensor(Training.iloc[:,conf.InputSIZE:].values, dtype=torch.float32)
	assert X_train.device.type == device.type
	assert y_train.device.type == device.type

	X_valid = torch.tensor(Validation.iloc[:,:conf.InputSIZE].values, dtype=torch.float32)
	y_valid = torch.tensor(Validation.iloc[:,conf.InputSIZE:].values, dtype=torch.float32)
	assert X_valid.device.type == device.type
	assert y_valid.device.type == device.type

	X_test  = torch.tensor(Test.iloc[:,:conf.InputSIZE].values, dtype=torch.float32)
	y_test  = torch.tensor(Test.iloc[:,conf.InputSIZE:].values, dtype=torch.float32)
	assert X_test.device.type == device.type
	assert y_test.device.type == device.type


	stats = {
		'OriginalInputs' : X_train.shape[0],
		'TrainingTIME' : [],
		'OpportunisticSeachTIME' : [],
		'FormalVerificationTIME' : [],
		'RepairTIME' : [],
		'SamplingTIME' : [],
		'LossOnValidation' : [],
		'LossOnTest' : [],
		'OpportunisticCEX' : [],
		'FormalVerificationCEX' : [],
		'NewPairs' : [],
		'TotalTIME' : None
	}

	loss_fn = training.custom_MSELoss
	lossT = None
	cegis_iteration = 0

	network = neural_network(args.hidden_layers, args.neurons_per_hidden_layer)
	
	while True:				
		cegis_iteration += 1	
		# print("iteration={}".format(cegis_iteration))
		if cegis_iteration > 1 and (not args.keep):
			network = neural_network(args.hidden_layers, args.neurons_per_hidden_layer) # new one
		
		soft = args.property if args.soft_acceleration else None
		
		network, lossV, TrainingTIME = training.train(network, X_train, y_train, X_valid, y_valid, args)			

		stats['TrainingTIME'].append(TrainingTIME)
		stats['LossOnValidation'].append(lossV)

		# compute loss on test
		with torch.no_grad():
			y_pred = network(X_test)
			lossT = loss_fn(y_pred, y_test).item()
			#print(lossT)
			stats['LossOnTest'].append(lossT)

		stats['OpportunisticSeachTIME'].append(0)
		stats['OpportunisticCEX'].append(0)
		stats['FormalVerificationTIME'].append(0)
		stats['FormalVerificationCEX'].append(0)
		stats['RepairTIME'].append(0)
		stats['SamplingTIME'].append(0)
		stats['NewPairs'].append(0)
		stats['CEGISiterations'] = cegis_iteration

		NumberOfCounterExamples = 0
		if args.counterexamples > 0:
			CEX, CEY, OpportunisticTime = fm.opportunistic_search_proxy(network, args, N=1000)
			NumberOfCounterExamples = CEX.shape[0] if isinstance(CEX, torch.Tensor) else sum(cex.shape[0] for cex in CEX)
			stats['OpportunisticSeachTIME'][-1] = OpportunisticTime
			stats['OpportunisticCEX'][-1] = NumberOfCounterExamples

			if NumberOfCounterExamples == 0:
				CEX, CEY, FormalVerificationTIME = fm.formal_verification_proxy(network, args)
				NumberOfCounterExamples = CEX.shape[0] if isinstance(CEX, torch.Tensor) else sum(cex.shape[0] for cex in CEX)
				stats['FormalVerificationTIME'][-1] = FormalVerificationTIME
				stats['FormalVerificationCEX'][-1] =  NumberOfCounterExamples
				
			if NumberOfCounterExamples > 0:
				REPX, REPY, RepairTIME = fm.repair_proxy(CEX, CEY, args)
				SX, SY, SamplingTIME = fm.sampling_proxy(REPX, REPY, args)
				X_train = torch.cat((X_train, REPX, SX), dim=0) if isinstance(REPX, torch.Tensor) else torch.cat(tuple([X_train] + REPX + SX), dim=0)
				y_train = torch.cat((y_train, REPY, SY), dim=0) if isinstance(REPX, torch.Tensor) else torch.cat(tuple([y_train] + REPY + SY), dim=0)
				stats['RepairTIME'][-1] = RepairTIME
				stats['NewPairs'][-1] = CEX.shape[0] + SX.shape[0] if isinstance(CEX, torch.Tensor) else sum(cex.shape[0] for cex in CEX) + sum(sx.shape[0] for sx in SX)
				stats['SamplingTIME'][-1] = SamplingTIME

		# statistics
		with open('experiments/{}/stats.json'.format(EXPERIMENT), 'w') as f:
			json.dump(stats, f)

		torch.save(network.state_dict(), 'experiments/{}/network-{}.pt'.format(EXPERIMENT,cegis_iteration))
		
		if NumberOfCounterExamples == 0:
			#print("No counterexamples found")
			break

	stats['TotalTIME'] = time.time() - START
	
	# final statistics
	with open('experiments/{}/stats.json'.format(EXPERIMENT), 'w') as f:
		json.dump(stats, f)