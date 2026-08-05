# ====================================================================
# Property 9
# ====================================================================
# Even if the previous advisory was "weak right", the presence of a nearby 
# intruder will cause the networks to output a "strong left" advisory instead
# ============================================================================
# Delta strenghtening
# Input: 2000 <= rho <= 7000, -0.4 <= theta <= -0.14, -3.141592 <= psi <= -3.141592+0.01, 100 <= vown <= 150, 0 <= vint <= 150
# Output: SL is minimal: (SL + DELTA <= COC) and (SL + DELTA <= WL) and (SL + DELTA <= WR) and (SL + DELTA <= SR)
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
	#print("precondition_enabling_batch P9")
	low  = torch.tensor(norm.Input_LB)
	high = torch.tensor(norm.Input_UB)
	
	low[conf.rho]    = norm.normalize_rho(2000)
	high[conf.rho]   = norm.normalize_rho(7000)

	low[conf.theta]  = norm.normalize_theta(-0.4)
	high[conf.theta] = norm.normalize_theta(-0.14)
	
	low[conf.psi]    = norm.normalize_psi(-3.141592)
	high[conf.psi]   = norm.normalize_psi(-3.141592+0.01)
	
	low[conf.vown]   = norm.normalize_vown(100)
	high[conf.vown]  = norm.normalize_vown(150)
	
	low[conf.vint]   = norm.normalize_vint(100)
	high[conf.vint]  = norm.normalize_vint(150)
	x = low + (high - low) * torch.rand(N, low.shape[0])
	return x

def precondition_mask(x):
	return (
			(x[:,conf.rho] >= norm.normalize_rho(2000)) & 
			(x[:,conf.rho] <= norm.normalize_rho(7000)) &

			(x[:,conf.theta] >= norm.normalize_theta(-0.4)) & 
			(x[:,conf.theta] <= norm.normalize_theta(-0.14)) & 

			(x[:,conf.psi] >= norm.normalize_psi(-3.141592)) &
			(x[:,conf.psi] <= norm.normalize_psi(-3.141592+0.01)) & 
			
			(x[:,conf.vown] >= norm.normalize_vown(100)) & 
			(x[:,conf.vown] <= norm.normalize_vown(150)) & 
			
			(x[:,conf.vint] >= norm.normalize_vint(100)) &
			(x[:,conf.vint] <= norm.normalize_vint(150))
			)

def postcondition_mask(x,y,args):
	return y[:, conf.sl]  - y[:, conf.ActionsDifferentFromSL].min(dim=1).values <= -args.delta
	
def soft_loss(network, N, args):	
	#print("soft_loss P9")
	x = precondition_enabling_batch(N)
	y = network(x)
	VS = torch.relu(y[:, conf.sl].unsqueeze(1) - y[:, conf.ActionsDifferentFromSL] + args.delta).sum(dim=1)
	mask = VS > 0
	return VS[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)

def opportunistic_search(network, args, N):
	#print("opportunistic_search P9")
	start = time.time()
	x = precondition_enabling_batch(N)
	with torch.inference_mode():
		y = network(x)

	mask = y[:, conf.sl]  - y[:, conf.ActionsDifferentFromSL].min(dim=1).values >= (args.epsilon - args.delta)
	CEX = x[mask][:args.counterexamples]
	CEY = y[mask][:args.counterexamples]
	OpportunisticSeachTIME = time.time() - start
	return CEX, CEY, OpportunisticSeachTIME

def MIQP_repair_constraints(m, x, y, s, args):
	#print("MIQP_repair_constraints P9")
	m.addConstr(y[conf.sl].item() + s[conf.sl] + args.delta <= y[conf.coc].item() + s[conf.coc])
	m.addConstr(y[conf.sl].item() + s[conf.sl] + args.delta <= y[conf.wl].item()  + s[conf.wl])
	m.addConstr(y[conf.sl].item() + s[conf.sl] + args.delta <= y[conf.wr].item()  + s[conf.wr])
	m.addConstr(y[conf.sl].item() + s[conf.sl] + args.delta <= y[conf.sr].item()  + s[conf.sr])

def repair(CEX, CEY, args):
	#print("repair P9")
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
	#print("sampling P9")
	start = time.time()
	N = REPX.shape[0] # actual counterexamples found
	x = REPX.repeat_interleave(args.sampling,0)
	y = REPY.repeat_interleave(args.sampling,0)

	x += 0.01 * torch.randn(args.sampling * N,conf.InputSIZE)
	y += 0.01 * torch.randn(args.sampling * N,conf.OutputSIZE)

	mask = ~precondition_mask(x) | postcondition_mask(x,y,args)
	return x[mask], y[mask], time.time() - start

def formal_verification(network, args):
	#print("formal_verification P9")
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

	Q.setLowerBound(X[conf.rho], norm.normalize_rho(2000))
	Q.setUpperBound(X[conf.rho], norm.normalize_rho(7000))
	Q.setLowerBound(X[conf.theta], norm.normalize_theta(-0.4))
	Q.setUpperBound(X[conf.theta], norm.normalize_theta(-0.14))
	Q.setLowerBound(X[conf.psi], norm.normalize_psi(-3.141592))
	Q.setUpperBound(X[conf.psi], norm.normalize_psi(-3.141592+0.01))
	Q.setLowerBound(X[conf.vown], norm.normalize_vown(100))
	Q.setUpperBound(X[conf.vown], norm.normalize_vown(150))
	Q.setLowerBound(X[conf.vint], norm.normalize_vint(100))
	Q.setUpperBound(X[conf.vint], norm.normalize_vint(150))
	
	# bigOR (SL - A >= epsilon - delta)
	disjunction = []
	for act in conf.ActionsDifferentFromSL:
		eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
		eq.addAddend(1, Y[conf.sl])
		eq.addAddend(-1, Y[act])
		eq.setScalar(args.epsilon - args.delta)
		disjunction.append([eq])
	Q.addDisjunctionConstraint(disjunction)

	options = Marabou.createOptions(numWorkers=args.cores, verbosity=0)
	sat, vals, stats = Q.solve(verbose=False,options=options)
	CEX = torch.tensor([[vals[x] for x in X]]) if sat == 'sat' else torch.empty(0,conf.InputSIZE)
	CEY = torch.tensor([[vals[y] for y in Y]]) if sat == 'sat' else torch.empty(0,conf.OutputSIZE)
	FormalVerificationTIME = time.time() - start
	return CEX, CEY, FormalVerificationTIME