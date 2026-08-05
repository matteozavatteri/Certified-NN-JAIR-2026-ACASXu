import conf
import normalization as norm

ACASXU = [
	'ACASXU_run2a_1_1_batch_2000',
	'ACASXU_run2a_1_2_batch_2000',
	'ACASXU_run2a_1_3_batch_2000',
	'ACASXU_run2a_1_4_batch_2000',
	'ACASXU_run2a_1_5_batch_2000',
	'ACASXU_run2a_1_6_batch_2000',
	'ACASXU_run2a_1_7_batch_2000',
	'ACASXU_run2a_1_8_batch_2000',
	'ACASXU_run2a_1_9_batch_2000',
	'ACASXU_run2a_2_1_batch_2000',
	'ACASXU_run2a_2_2_batch_2000',
	'ACASXU_run2a_2_3_batch_2000',
	'ACASXU_run2a_2_4_batch_2000',
	'ACASXU_run2a_2_5_batch_2000',
	'ACASXU_run2a_2_6_batch_2000',
	'ACASXU_run2a_2_7_batch_2000',
	'ACASXU_run2a_2_8_batch_2000',
	'ACASXU_run2a_2_9_batch_2000',
	'ACASXU_run2a_3_1_batch_2000',
	'ACASXU_run2a_3_2_batch_2000',
	'ACASXU_run2a_3_3_batch_2000',
	'ACASXU_run2a_3_4_batch_2000',
	'ACASXU_run2a_3_5_batch_2000',
	'ACASXU_run2a_3_6_batch_2000',
	'ACASXU_run2a_3_7_batch_2000',
	'ACASXU_run2a_3_8_batch_2000',
	'ACASXU_run2a_3_9_batch_2000',
	'ACASXU_run2a_4_1_batch_2000',
	'ACASXU_run2a_4_2_batch_2000',
	'ACASXU_run2a_4_3_batch_2000',
	'ACASXU_run2a_4_4_batch_2000',
	'ACASXU_run2a_4_5_batch_2000',
	'ACASXU_run2a_4_6_batch_2000',
	'ACASXU_run2a_4_7_batch_2000',
	'ACASXU_run2a_4_8_batch_2000',
	'ACASXU_run2a_4_9_batch_2000',
	'ACASXU_run2a_5_1_batch_2000',
	'ACASXU_run2a_5_2_batch_2000',
	'ACASXU_run2a_5_3_batch_2000',
	'ACASXU_run2a_5_4_batch_2000',
	'ACASXU_run2a_5_5_batch_2000',
	'ACASXU_run2a_5_6_batch_2000',
	'ACASXU_run2a_5_7_batch_2000',
	'ACASXU_run2a_5_8_batch_2000',
	'ACASXU_run2a_5_9_batch_2000'
	]

#acasxu_nnets = acasxu_nnets[:1]

from maraboupy import Marabou
import onnxruntime as ort
import numpy as np

import os
import pandas as pd

if not os.path.exists("datasets"):
	os.mkdir("datasets")

# Uniform sampling
N = 10000
low = np.array(norm.Input_LB)
high = np.array(norm.Input_UB)
X = low + (high - low) * np.random.rand(N, low.shape[0])
X = X.astype(np.float32)    
#print(X)
Y = np.empty((N, conf.OutputSIZE))
#print(Y)

column_names = ['rho','theta','psi','vown','vint','COC','WL','WR','SL','SR']

# generate datasets
for net in ACASXU:
	print('Generating {}.csv ...'.format(net), end='')
	onnx_vnncomp  = '{}.onnx'.format(net)
	sessVNNCOMP = ort.InferenceSession('vnncomp/{}'.format(onnx_vnncomp))
	for i in range(len(X)):
		y = sessVNNCOMP.run(None, {sessVNNCOMP.get_inputs()[0].name: [[[X[i]]]]})[0][0]
		Y[i,] = y
	XY = np.concatenate((X, Y), axis=1)
	
	df = pd.DataFrame(XY, columns=column_names)
	df.to_csv('datasets/{}.csv'.format(net),index=False)
	print('Done.')








