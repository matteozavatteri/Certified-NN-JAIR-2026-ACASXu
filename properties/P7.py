# ====================================================================
# Property 7
# ====================================================================
# If vertical separation is large, the network will never advise a strong turn
# ====================================================================
# Delta strenghtening
# Input: 
# 0 <= rho <= 60760, 
# -3.141592 <= theta <= 3.141592
# -3.141592 <= psi <= 3.141592 
# 100 <= vown <= 1200, 
# 0 <= vint <= 1200
# Output: the output scores for SL and SR are never minimal
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
	#print("precondition_enabling_batch P7")
	low  = torch.tensor(norm.Input_LB)
	high = torch.tensor(norm.Input_UB)
	
	low[conf.rho]    = norm.normalize_rho(0)
	high[conf.rho]   = norm.normalize_rho(60760)

	low[conf.theta]  = norm.normalize_theta(-3.141592)
	high[conf.theta] = norm.normalize_theta(3.141592)
	
	low[conf.psi]    = norm.normalize_psi(-3.141592)
	high[conf.psi]   = norm.normalize_psi(3.141592)
	
	low[conf.vown]   = norm.normalize_vown(100)
	high[conf.vown]  = norm.normalize_vown(1200)
	
	low[conf.vint]   = norm.normalize_vint(0)
	high[conf.vint]  = norm.normalize_vint(1200)

	x = low + (high - low) * torch.rand(N, low.shape[0])
	return x

def precondition_mask(x):
	return (
			(x[:,conf.rho] >= norm.normalize_rho(0)) & 
			(x[:,conf.rho] <= norm.normalize_rho(60760)) &

			(x[:,conf.theta] >= norm.normalize_theta(-3.141592)) & 
			(x[:,conf.theta] <= norm.normalize_theta(3.141592)) & 

			(x[:,conf.psi] >= norm.normalize_psi(-3.141592)) &
			(x[:,conf.psi] <= norm.normalize_psi(3.141592)) & 
			
			(x[:,conf.vown] >= norm.normalize_vown(100)) & 
			(x[:,conf.vown] <= norm.normalize_vown(1200)) & 
			
			(x[:,conf.vint] >= norm.normalize_vint(0)) &
			(x[:,conf.vint] <= norm.normalize_vint(1200))
			)

def postcondition_mask(x,y,args):
	return (
		(y[:, conf.sl] - y[:, conf.ActionsDifferentFromSL_SR].min(dim=1).values >= args.delta) &
		(y[:, conf.sr] - y[:, conf.ActionsDifferentFromSL_SR].min(dim=1).values >= args.delta)
		)

def soft_loss(network, N, args):	
	#print("soft_loss P7")
	x = precondition_enabling_batch(N)
	y = network(x)
	VS =  torch.relu(y[:, conf.ActionsDifferentFromSL_SR] - y[:, conf.sl].unsqueeze(1) + args.delta).prod(dim=1) + torch.relu(y[:, conf.ActionsDifferentFromSL_SR] - y[:, conf.sr].unsqueeze(1) + args.delta).prod(dim=1)
	mask = VS > 0
	return VS[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)

def opportunistic_search(network, args, N):
	#print("opportunistic_search P7")
	start = time.time()
	x = precondition_enabling_batch(N)
	with torch.inference_mode():
		y = network(x)

	mask = (y[:, conf.sl] - y[:, conf.ActionsDifferentFromSL_SR].min(dim=1).values <= (args.delta - args.epsilon)) | (y[:, conf.sr] - y[:, conf.ActionsDifferentFromSL_SR].min(dim=1).values <= (args.delta - args.epsilon))
	CEX = x[mask][:args.counterexamples]
	CEY = y[mask][:args.counterexamples]
	OpportunisticSeachTIME = time.time() - start
	return CEX, CEY, OpportunisticSeachTIME

def MIQP_repair_constraints(m, x, y, s, args):
	#print("MIQP_repair_constraints P7")
	a1 = m.addVar(vtype=gp.GRB.BINARY)
	a2 = m.addVar(vtype=gp.GRB.BINARY)
	a3 = m.addVar(vtype=gp.GRB.BINARY)
	m.addGenConstrIndicator(a1, True, y[conf.sr].item() + s[conf.sr] >= y[conf.coc].item() + s[conf.coc] + args.delta)
	m.addGenConstrIndicator(a2, True, y[conf.sr].item() + s[conf.sr] >= y[conf.wl].item()  + s[conf.wl] + args.delta)
	m.addGenConstrIndicator(a3, True, y[conf.sr].item() + s[conf.sr] >= y[conf.wr].item()  + s[conf.wr] + args.delta)
	m.addConstr(a1 + a2 + a3>= 1)	

	b1 = m.addVar(vtype=gp.GRB.BINARY)
	b2 = m.addVar(vtype=gp.GRB.BINARY)
	b3 = m.addVar(vtype=gp.GRB.BINARY)
	m.addGenConstrIndicator(b1, True, y[conf.sl].item() + s[conf.sl] >= y[conf.coc].item() + s[conf.coc] + args.delta)
	m.addGenConstrIndicator(b2, True, y[conf.sl].item() + s[conf.sl] >= y[conf.wl].item()  + s[conf.wl] + args.delta)
	m.addGenConstrIndicator(b3, True, y[conf.sl].item() + s[conf.sl] >= y[conf.wr].item()  + s[conf.wr] + args.delta)
	m.addConstr(b1 + b2 + b3 >= 1)	

def repair(CEX, CEY, args):
	#print("repair P7")
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
	#print("sampling P7")
	start = time.time()
	N = REPX.shape[0] # actual counterexamples found
	x = REPX.repeat_interleave(args.sampling,0)
	y = REPY.repeat_interleave(args.sampling,0)

	x += 0.01 * torch.randn(args.sampling * N,conf.InputSIZE)
	y += 0.01 * torch.randn(args.sampling * N,conf.OutputSIZE)

	mask = ~precondition_mask(x) | postcondition_mask(x,y,args)
	return x[mask], y[mask], time.time() - start

def formal_verification(network, args):
	#print("formal_verification P7")
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

	Q.setLowerBound(X[conf.rho],  norm.normalize_rho(0))
	Q.setUpperBound(X[conf.rho],  norm.normalize_rho(60760))
	Q.setLowerBound(X[conf.theta], norm.normalize_theta(-3.141592))
	Q.setUpperBound(X[conf.theta], norm.normalize_theta(3.141592))
	Q.setLowerBound(X[conf.psi], norm.normalize_psi(-3.141592)) 
	Q.setUpperBound(X[conf.psi], norm.normalize_psi(3.141592))  
	Q.setLowerBound(X[conf.vown], norm.normalize_vown(100)) 
	Q.setUpperBound(X[conf.vown], norm.normalize_vown(1200))  
	Q.setLowerBound(X[conf.vint], norm.normalize_vint(0)) 
	Q.setUpperBound(X[conf.vint], norm.normalize_vint(1200))  

	# [bigAND (SL - A <= delta - epsilon)] or [bigAND (SR - A <= delta - epsilon)]
	disjunction = []

	conjSL = []
	for act in conf.ActionsDifferentFromSL_SR:
		eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
		eq.addAddend(1, Y[conf.sl])
		eq.addAddend(-1, Y[act])
		eq.setScalar(args.delta - args.epsilon)
		conjSL.append(eq)
	disjunction.append(conjSL)

	conjSR = []
	for act in conf.ActionsDifferentFromSL_SR:
		eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
		eq.addAddend(1, Y[conf.sr])
		eq.addAddend(-1, Y[act])
		eq.setScalar(args.delta - args.epsilon)
		conjSR.append(eq)
	disjunction.append(conjSR)

	Q.addDisjunctionConstraint(disjunction)

	options = Marabou.createOptions(numWorkers=args.cores, verbosity=0)
	sat, vals, stats = Q.solve(verbose=False,options=options)
	CEX = torch.tensor([[vals[x] for x in X]]) if sat == 'sat' else torch.empty(0,conf.InputSIZE)
	CEY = torch.tensor([[vals[y] for y in Y]]) if sat == 'sat' else torch.empty(0,conf.OutputSIZE)
	FormalVerificationTIME = time.time() - start
	return CEX, CEY, FormalVerificationTIME