# ====================================================================
# Property 3
# ====================================================================
# If the intruder is directly ahead and is moving towards the ownship, the score 
# of a COC advisory will not be minimal
# --------------------------------------------------------------------
# Delta strenghtening
# Input: 1500 <= rho <= 1800, -0.06 <= theta <= 0.06, psi >= 3.10, vown >= 980, vint >= 960
# Output: COC is not minimal: (COC >= WL + DELTA) or (COC >= WR + DELTA) or (COC >= SL + DELTA) or (COC >= SR + DELTA)
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
	#print("precondition_enabling_batch P3")
	low  = torch.tensor(norm.Input_LB)
	high = torch.tensor(norm.Input_UB)
	
	low[conf.rho]    = norm.normalize_rho(1500)
	high[conf.rho]   = norm.normalize_rho(1800)
	
	low[conf.theta]  = norm.normalize_theta(-0.6)
	high[conf.theta] = norm.normalize_theta(0.6)
	
	low[conf.psi]    = norm.normalize_psi(3.10)
	
	low[conf.vown]   = norm.normalize_vown(980)
	low[conf.vint]   = norm.normalize_vint(960)

	x = low + (high - low) * torch.rand(N, low.shape[0])
	return x

def precondition_mask(x):
	return (
		(x[:,conf.rho] >= norm.normalize_rho(1500)) & 
		(x[:,conf.rho] <= norm.normalize_rho(1800)) & 
		(x[:,conf.theta] >= norm.normalize_theta(-0.06)) & 
		(x[:,conf.theta] <= norm.normalize_theta(0.06)) & 
		(x[:,conf.psi] >= norm.normalize_psi(3.10)) & 
		(x[:,conf.vown] >= norm.normalize_vown(980)) & 
		(x[:,conf.vint] >= norm.normalize_vint(960))
	)

def postcondition_mask(x,y,args):
	return y[:, conf.coc] - y[:, conf.ActionsDifferentFromCOC].min(dim=1).values >= args.delta

def soft_loss(network, N, args):	
	#print("soft_loss P3")
	x = precondition_enabling_batch(N)
	y = network(x)
	VS = torch.relu(y[:, conf.ActionsDifferentFromCOC] + args.delta - y[:, conf.coc].unsqueeze(1)).prod(dim=1)
	mask = VS > 0
	return VS[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)

def opportunistic_search(network, args, N):
	#print("opportunistic_search P3")
	start = time.time()
	x = precondition_enabling_batch(N)
	with torch.inference_mode():
		y = network(x)

	mask = y[:, conf.coc] - y[:, conf.ActionsDifferentFromCOC].min(dim=1).values <= (args.delta - args.epsilon)
	CEX = x[mask][:args.counterexamples]
	CEY = y[mask][:args.counterexamples]
	OpportunisticSeachTIME = time.time() - start
	return CEX, CEY, OpportunisticSeachTIME

def MIQP_repair_constraints(m, x, y, s, args):
	#print("MIQP_repair_constraints P3")
	dWL = m.addVar(vtype=gp.GRB.BINARY)
	dWR = m.addVar(vtype=gp.GRB.BINARY)
	dSL = m.addVar(vtype=gp.GRB.BINARY)
	dSR = m.addVar(vtype=gp.GRB.BINARY)
	m.addGenConstrIndicator(dWL, True, y[conf.coc].item() + s[conf.coc] >= y[conf.wl].item() + s[conf.wl] + args.delta)
	m.addGenConstrIndicator(dWR, True, y[conf.coc].item() + s[conf.coc] >= y[conf.wr].item() + s[conf.wr] + args.delta)
	m.addGenConstrIndicator(dSL, True, y[conf.coc].item() + s[conf.coc] >= y[conf.sl].item() + s[conf.sl] + args.delta)
	m.addGenConstrIndicator(dSR, True, y[conf.coc].item() + s[conf.coc] >= y[conf.sr].item() + s[conf.sr] + args.delta)
	m.addConstr(dWL + dWR + dSL + dSR >= 1)


def repair(CEX, CEY, args):
	#print("repair P3")
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
	#print("sampling P3")
	start = time.time()
	N = REPX.shape[0] # actual counterexamples found
	x = REPX.repeat_interleave(args.sampling,0)
	y = REPY.repeat_interleave(args.sampling,0)

	x += 0.01 * torch.randn(args.sampling * N,conf.InputSIZE)
	y += 0.01 * torch.randn(args.sampling * N,conf.OutputSIZE)

	mask = ~precondition_mask(x) | postcondition_mask(x,y,args)
	return x[mask], y[mask], time.time() - start

def formal_verification(network, args):
	#print("formal_verification P3")
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
	
	Q.setLowerBound(X[conf.rho], norm.normalize_rho(1500))
	Q.setUpperBound(X[conf.rho], norm.normalize_rho(1800))	

	Q.setLowerBound(X[conf.theta], norm.normalize_theta(-0.6))
	Q.setUpperBound(X[conf.theta], norm.normalize_theta(0.6))	

	Q.setLowerBound(X[conf.psi], norm.normalize_psi(3.10))

	Q.setLowerBound(X[conf.vown], norm.normalize_vown(980))
	Q.setLowerBound(X[conf.vint], norm.normalize_vint(960))	
	
	# bigAND (A - COC <= DELTA - epsilon)
	for act in conf.ActionsDifferentFromCOC:
		eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
		eq.addAddend(1, Y[conf.coc])
		eq.addAddend(-1, Y[act])
		eq.setScalar(args.delta - args.epsilon)
		Q.addEquation(eq)

	options = Marabou.createOptions(numWorkers=args.cores, verbosity=0)
	sat, vals, stats = Q.solve(verbose=False,options=options)
	CEX = torch.tensor([[vals[x] for x in X]]) if sat == 'sat' else torch.empty(0,conf.InputSIZE)
	CEY = torch.tensor([[vals[y] for y in Y]]) if sat == 'sat' else torch.empty(0,conf.OutputSIZE)
	FormalVerificationTIME = time.time() - start
	return CEX, CEY, FormalVerificationTIME