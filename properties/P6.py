# ====================================================================
# Property 6
# ====================================================================
# If the intruder is sufficiently far away, the network advises COC.
# ====================================================================
# Delta strenghtening
# Input: 
# 12000 <= rho <= 62000, 
# (0.7 <= theta <= 3.141592) or (-3.141592 <= theta <= -0.7), 
# -3.141592 <= psi <= 3.141592 + 0.005, 
# 100 <= vown <= 1200, 
# 0 <= vint <= 1200
# Output: COC is minimal: (COC + DELTA <= WL) and (COC + DELTA <= WR) and (COC + DELTA <= SL) and (COC + DELTA <= SR)
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
	#print("precondition_enabling_batch P6")
	low  = torch.tensor(norm.Input_LB)
	high = torch.tensor(norm.Input_UB)
	
	low[conf.rho]    = norm.normalize_rho(12000)
	high[conf.rho]   = norm.normalize_rho(62000)

	low[conf.theta]  = norm.normalize_theta(0.7)
	high[conf.theta] = norm.normalize_theta(3.141592)
	
	low[conf.psi]    = norm.normalize_psi(-3.141592)
	high[conf.psi]   = norm.normalize_psi(-3.141592+0.005)
	
	low[conf.vown]   = norm.normalize_vown(100)
	high[conf.vown]  = norm.normalize_vown(1200)
	
	low[conf.vint]   = norm.normalize_vint(0)
	high[conf.vint]  = norm.normalize_vint(1200)

	x1 = low + (high - low) * torch.rand(int(N/2), low.shape[0])
	
	low[conf.theta]  = norm.normalize_theta(-3.141592)
	high[conf.theta] = norm.normalize_theta(-0.7)
	
	x2 = low + (high - low) * torch.rand(int(N/2), low.shape[0])

	return torch.cat((x1,x2),dim=0)

def precondition_mask(x):
	return (
		(x[:,conf.rho] >= norm.normalize_rho(12000)) & 
		(x[:,conf.rho] <= norm.normalize_rho(62000)) & 

		(
			((x[:,conf.theta] >= norm.normalize_theta(0.7))        & (x[:,conf.theta] <= norm.normalize_theta(3.141592))) | 
			((x[:,conf.theta] >= norm.normalize_theta(-3.141592) ) & (x[:,conf.theta] <= norm.normalize_theta(-0.7)))
		) & 

		(x[:,conf.psi] >= norm.normalize_psi(-3.141592)) &
		(x[:,conf.psi] <= norm.normalize_psi(-3.141592+0.005)) & 
			
		(x[:,conf.vown] >= norm.normalize_vown(100)) & 
		(x[:,conf.vown] <= norm.normalize_vown(1200)) & 
			
		(x[:,conf.vint] >= norm.normalize_vint(0)) &
		(x[:,conf.vint] <= norm.normalize_vint(1200))
		)

def postcondition_mask(x,y,args):
	return y[:, conf.coc] - y[:, conf.ActionsDifferentFromCOC].min(dim=1).values <= -args.delta

def soft_loss(network, N, args):	
	#print("soft_loss P6")
	x = precondition_enabling_batch(N)
	y = network(x)
	VS = torch.relu(y[:, conf.coc].unsqueeze(1) - y[:, conf.ActionsDifferentFromCOC] + args.delta).sum(dim=1)
	mask = VS > 0
	return VS[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)

def opportunistic_search(network, args, N):
	#print("opportunistic_search P6")
	start = time.time()
	x = precondition_enabling_batch(N)
	with torch.inference_mode():
		y = network(x)

	mask = y[:, conf.coc] - y[:, conf.ActionsDifferentFromCOC].min(dim=1).values >= (args.epsilon - args.delta)
	CEX = x[mask][:args.counterexamples]
	CEY = y[mask][:args.counterexamples]
	OpportunisticSeachTIME = time.time() - start
	return CEX, CEY, OpportunisticSeachTIME

def MIQP_repair_constraints(m, x, y, s, args):
	#print("MIQP_repair_constraints P6")
	m.addConstr(y[conf.coc].item() + s[conf.coc] + args.delta <= y[conf.wl].item() + s[conf.wl])
	m.addConstr(y[conf.coc].item() + s[conf.coc] + args.delta <= y[conf.wr].item() + s[conf.wr])
	m.addConstr(y[conf.coc].item() + s[conf.coc] + args.delta <= y[conf.sl].item() + s[conf.sl])
	m.addConstr(y[conf.coc].item() + s[conf.coc] + args.delta <= y[conf.sr].item() + s[conf.sr])
		

def repair(CEX, CEY, args):
	#print("repair P6")
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
	#print("sampling P6")
	start = time.time()
	N = REPX.shape[0] # actual counterexamples found
	x = REPX.repeat_interleave(args.sampling,0)
	y = REPY.repeat_interleave(args.sampling,0)

	x += 0.01 * torch.randn(args.sampling * N,conf.InputSIZE)
	y += 0.01 * torch.randn(args.sampling * N,conf.OutputSIZE)

	mask = ~precondition_mask(x) | postcondition_mask(x,y,args)
	return x[mask], y[mask], time.time() - start

def formal_verification(network, args):
	#print("formal_verification P6")
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

	Q.setLowerBound(X[conf.rho], norm.normalize_rho(12000))
	Q.setUpperBound(X[conf.rho], norm.normalize_rho(62000))	
	
	Q.setLowerBound(X[conf.psi], norm.normalize_psi(-3.141592))
	Q.setUpperBound(X[conf.psi], norm.normalize_psi(-3.141592+0.005))
	
	Q.setLowerBound(X[conf.vown], norm.normalize_vown(100))
	Q.setUpperBound(X[conf.vown], norm.normalize_vown(1200))
	
	Q.setLowerBound(X[conf.vint], norm.normalize_vint(0))
	Q.setUpperBound(X[conf.vint], norm.normalize_vint(1200))	
	
	
	disjunction = []
	LB1 = MarabouUtils.Equation(MarabouCore.Equation.GE)
	LB1.addAddend(1, X[conf.theta])
	LB1.setScalar(norm.normalize_theta(0.7))

	UB1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
	UB1.addAddend(1, X[conf.theta])
	UB1.setScalar(norm.normalize_theta(3.141592))

	disjunction.append([LB1, UB1])

	LB2 = MarabouUtils.Equation(MarabouCore.Equation.GE)
	LB2.addAddend(1, X[conf.theta])
	LB2.setScalar(norm.normalize_theta(-3.141592))

	UB2 = MarabouUtils.Equation(MarabouCore.Equation.LE)
	UB2.addAddend(1, X[conf.theta])
	UB2.setScalar(norm.normalize_theta(-0.7))

	disjunction.append([LB2, UB2])

	Q.addDisjunctionConstraint(disjunction)
	
	# bigOR (COC + ACT >= EPSILON - DELTA) 
	disjunction = []
	for act in conf.ActionsDifferentFromCOC:
		eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
		eq.addAddend(1, Y[conf.coc])
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