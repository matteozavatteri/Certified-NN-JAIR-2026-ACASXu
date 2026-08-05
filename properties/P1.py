# ====================================================================
# Property 1
# ====================================================================
# If the intruder is distant and is significantly slower than the ownship,
# the score of a COC advisory will always below a certain fixed threshold.
# --------------------------------------------------------------------
# Delta strenghtening
# Input: rho >= 55947.691, vown >= 1145, vint <= 60
# Output: coc + DELTA <= 1500
# ====================================================================

import time
import random
from maraboupy import Marabou, MarabouCore
from maraboupy import MarabouUtils
import tempfile
import gurobipy as gp

import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

import conf
import normalization as norm

def precondition_enabling_batch(N):
	print("precondition_enabling_batch P1")
	low  = torch.tensor(norm.Input_LB)
	high = torch.tensor(norm.Input_UB)

	low[conf.rho]   = norm.normalize_rho(55947.691)
	low[conf.vown]  = norm.normalize_vown(1145)
	high[conf.vint] = norm.normalize_vint(60)

	x = low + (high - low) * torch.rand(N, low.shape[0])
	return x

def precondition_mask(x):
	return (
		(x[:,conf.rho]  >= norm.normalize_rho(55947.691)) & 
		(x[:,conf.vown] >= norm.normalize_vown(1145)) & 
		(x[:,conf.vint] <= norm.normalize_vint(60))
		)

def postcondition_mask(x,y,args):
	return y[:, conf.coc] <= (norm.normalize_output(1500) - args.delta)

def soft_loss(network, N, args):	
	print("soft_loss P1")
	x = precondition_enabling_batch(N)
	y = network(x)
	VS = torch.relu(y[:, conf.coc] + (args.delta - norm.normalize_output(1500)))
	mask = VS > 0
	return VS[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)

def opportunistic_search(network, args, N):
	print("opportunistic_search P1")
	start = time.time()
	x = precondition_enabling_batch(N)
	with torch.inference_mode():
		y = network(x)

	mask = y[:, conf.coc] >= norm.normalize_output(1500) + args.epsilon - args.delta

	CEX = x[mask][:args.counterexamples]
	CEY = y[mask][:args.counterexamples]
	OpportunisticSeachTIME = time.time() - start
	return CEX, CEY, OpportunisticSeachTIME

def MIQP_repair_constraints(m, x, y, s, args):
	print("MIQP_repair_constraints P1")
	m.addConstr(y[conf.coc].item() + s[conf.coc] <= (norm.normalize_output(1500) - args.delta))

def repair(CEX, CEY, args):
	#print("repair P1")
	start = time.time()
	env = gp.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()
	
	for i in range(CEY.shape[0]):
		m = gp.Model("repair",env=env)
		S = [m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for _ in range(conf.OutputSIZE)]
		MIQP_repair_constraints(m, CEX[i], CEY[i], S, args)
		m.setObjective(gp.quicksum(s ** 2 for s in S), gp.GRB.MINIMIZE)
		m.optimize()
		assert m.Status == gp.GRB.OPTIMAL

		for act in range(conf.OutputSIZE):
			CEY[i][act] += S[act].X
		
		m.dispose()
	RepairTIME = time.time() - start
	return CEX, CEY, RepairTIME


def sampling(REPX, REPY, args):
	print("sampling P1")
	start = time.time()
	N = REPX.shape[0] # actual counterexamples found
	x = REPX.repeat_interleave(args.sampling,0)
	y = REPY.repeat_interleave(args.sampling,0)

	x += 0.01 * torch.randn(args.sampling * N,conf.InputSIZE)
	y += 0.01 * torch.randn(args.sampling * N,conf.OutputSIZE)

	mask = ~precondition_mask(x) | postcondition_mask(x,y,args)
	return x[mask], y[mask], time.time() - start

def formal_verification(network, args):
	print("formal_verification P1")
	start = time.time()
	Q = None
	with tempfile.TemporaryDirectory() as tmpdir:
		network.eval()
		x = torch.zeros(1, conf.InputSIZE)
		onnx_path = os.path.join(tmpdir, "model.onnx")
		with torch.inference_mode():		
			torch.onnx.export(network, x, onnx_path)
		Q = Marabou.read_onnx(onnx_path)

	assert Q is not None

	X, Y = Q.inputVars[0][0], Q.outputVars[0][0]

	for i in range(conf.InputSIZE):
		Q.setLowerBound(X[i], norm.Input_LB[i])
		Q.setUpperBound(X[i], norm.Input_UB[i])
	
	Q.setLowerBound(X[conf.rho], norm.normalize_rho(55947.691))	
	Q.setLowerBound(X[conf.vown], norm.normalize_vown(1145))
	Q.setUpperBound(X[conf.vint], norm.normalize_vint(60))	

	
	# coc + DELTA >= 1500 + epsilon
	eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
	eq.addAddend(1, Y[conf.coc])
	eq.setScalar(norm.normalize_output(1500) + args.epsilon - args.delta)
	Q.addEquation(eq)

	options = Marabou.createOptions(numWorkers=args.cores, verbosity=0)
	sat, vals, stats = Q.solve(verbose=False,options=options)
	CEX = torch.tensor([[vals[x] for x in X]]) if sat == 'sat' else torch.empty(0,conf.InputSIZE)
	CEY = torch.tensor([[vals[y] for y in Y]]) if sat == 'sat' else torch.empty(0,conf.OutputSIZE)
	FormalVerificationTIME = time.time() - start
	return CEX, CEY, FormalVerificationTIME