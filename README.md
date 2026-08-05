# Paper

This repository contains the supplementary material of the paper:

	Matteo Zavatteri, Davide Bresolin, Nicolò Navarin. 
	Certified Neural Networks: from Verification to Synthesis.
	Journal of Artificial Intelligence Research, 2026.

# Dataset

This code works on the original dataset used in [VNN-COMP](https://vnn-comp.github.io). 

If you want to recreate those datasets:

- Download [https://github.com/VNN-COMP/vnncomp2025_benchmarks/tree/main/benchmarks/acasxu_2023/onnx](https://github.com/VNN-COMP/vnncomp2025_benchmarks/tree/main/benchmarks/acasxu_2023/onnx)
- Gzip them and put all of them in a folder called `vnncomp`
- Run `$ python3 preprocessing.py`. This will create the file `baseline.csv`.


To run all the experiments of the paper run:

	$ python3 Run.py

# Usage

To run a single experiment run 

	$ python3 ACASXuCertifier.py -h 

to see the help screen (see also inside Run.py to have example of possible commands).

# Data
The original data of the experimental evaluation discussed in the paper in contained in the folder `experiments`.
