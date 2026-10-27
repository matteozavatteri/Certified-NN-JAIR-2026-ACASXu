# Certified Neural Networks: from Verification to Synthesis — ACAS Xu experiments

This repository contains the code and data for the ACAS Xu experiments in:

> Matteo Zavatteri, Davide Bresolin, Nicolò Navarin.
> **Certified Neural Networks: from Verification to Synthesis.**
> *Journal of Artificial Intelligence Research*, 87, 2026.
> DOI: [10.1613/jair.1.22825](https://doi.org/10.1613/jair.1.22825)

```bibtex
@article{ZavatteriBresolinNavarin2026,
  author  = {Zavatteri, Matteo and Bresolin, Davide and Navarin, Nicol{\`o}},
  title   = {Certified Neural Networks: from Verification to Synthesis},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {87},
  year    = {2026},
  doi     = {10.1613/jair.1.22825},
  url     = {https://jair.org/index.php/jair/article/view/22825}
}
```

The paper was published in the JAIR special track [*Integration of Logical Constraints in Deep Learning*](https://www.jair.org/index.php/jair/SpecialTrack-LogicDL).

This repository includes:

- the training data (`datasets/`) and the script that generates it (`datasets.py`),
- the synthesis tool (`ACASXuCertifier.py` and `properties/`),
- the raw results of every run reported in the paper (`experiments/`),
- the scripts that turn those results into the paper's tables and figures, and the figures themselves (`img/`).

## What the code does

`ACASXuCertifier.py` trains a feed-forward ReLU network that imitates one of the 45 ACAS Xu networks. It then uses a counterexample-guided inductive synthesis (CEGIS) loop to make the network satisfy a given safety property. Each CEGIS iteration does the following:

1. **Train** a network on the current training set: Adam, batch size 100, early stopping on the validation loss, and an asymmetric MSE loss that penalises underestimating the score of the optimal advisory and overestimating the others (`training.py`).
2. **Opportunistic search**: evaluate the network on 1000 random inputs that satisfy the property's precondition and collect any that violate the postcondition.
3. **Formal verification**: if step 2 finds nothing, export the network to ONNX and ask [Marabou](https://github.com/NeuralNetworkVerification/Marabou) for a counterexample.
4. **Repair**: for each counterexample (at most `--counterexamples`), solve a small MIQP with Gurobi. It finds the smallest change to the network's output scores that makes the property hold.
5. **Sampling**: add `--sampling` noisy copies of each repaired point that still satisfy the property.
6. Add the repaired and sampled pairs to the training set and go back to step 1. By default each iteration trains a new network from scratch (use `--keep` to keep training the same one).

The loop stops when Marabou proves that no counterexample exists, so the last network saved is certified for the property.

With `--soft-acceleration` (`-SA` in result names), training also adds a penalty for property violations on random inputs from the precondition region. This usually reduces the number of CEGIS iterations.

The two tolerance parameters are:

- `--delta` (Δ): the margin by which the postcondition must hold. The property is strengthened by Δ.
- `--epsilon` (ε): how far a point must violate the Δ-strengthened postcondition to count as a counterexample. ε must be smaller than Δ.

## Dataset

The networks are trained to imitate the 45 ACAS Xu networks from:

> Kyle D. Julian, Jessica Lopez, Jeffrey S. Brush, Michael P. Owen, Mykel J. Kochenderfer.
> *Policy compression for aircraft collision avoidance systems.*
> Digital Avionics Systems Conference (DASC), 2016.

We use the ONNX versions distributed with [VNN-COMP](https://vnn-comp.github.io), benchmark [`acasxu_2023`](https://github.com/VNN-COMP/vnncomp2025_benchmarks/tree/main/benchmarks/acasxu_2023/onnx). There is one network per combination of:

- `pra`, the previous advisory: `1` COC, `2` WL, `3` WR, `4` SL, `5` SR;
- `tau`, the time to loss of vertical separation in seconds: `1` = 0, `2` = 1, `3` = 5, `4` = 10, `5` = 20, `6` = 40, `7` = 60, `8` = 80, `9` = 100.

That makes 5 × 9 = 45 networks. For each one, `datasets/ACASXU_run2a_<pra>_<tau>_batch_2000.csv` holds 10,000 inputs drawn uniformly from the normalised input box, together with the five scores that the original network gives them. All values are normalised:

| Inputs (`conf.py`) | Meaning | Range | Denormalisation |
|---|---|---|---|
| ρ (`rho`) | distance to the intruder | 0 – 60760 ft | 60261 · x + 19791.091 ft |
| θ (`theta`) | angle to the intruder | −π – π rad | 2π · x rad |
| ψ (`psi`) | heading of the intruder relative to the ownship | −π – π rad | 2π · x rad |
| v_own (`vown`) | speed of the ownship | 100 – 1200 ft/s | 1100 · x + 650 ft/s |
| v_int (`vint`) | speed of the intruder | 0 – 1200 ft/s | 1200 · x + 600 ft/s |

The outputs are scores for the five advisories: **COC** (clear of conflict), **WL**/**WR** (weak left/right) and **SL**/**SR** (strong left/right). An output score y denormalises to 373.94992 · y + 7.5188840201005975. The advisory with the **lowest** score is the one issued. The bounds and constants are in `normalization.py`.

After a seeded shuffle, the samples are split into 8,000 for training, 1,000 for validation and 1,000 for test.

### Regenerating the datasets

The CSV files are already in `datasets/`. To build them again:

1. Download the 45 `ACASXU_run2a_<pra>_<tau>_batch_2000.onnx` files from the VNN-COMP link above into a folder called `vnncomp/` in the repository root. If they come gzipped, decompress them (`gunzip vnncomp/*.gz`).
2. Run `python3 datasets.py`.

The script does not fix a random seed, so the new samples will differ from the ones in `datasets/`.

## Properties

Each property is in `properties/P<k>.py`. Each file defines the precondition, the postcondition, the soft loss, the opportunistic search, the Marabou query, the repair constraints and the sampling. `fm.py` selects the module that matches the property number. The properties are φ1–φ10 from:

> Guy Katz, Clark Barrett, David L. Dill, Kyle Julian, Mykel J. Kochenderfer.
> *Reluplex: An efficient SMT solver for verifying deep neural networks.*
> Computer Aided Verification (CAV), 2017.

| ID | Property | Datasets |
|---|---|---|
| $P_{1}$ | If the intruder is distant and much slower than the ownship, the COC score stays below a fixed threshold. | all |
| $P_{2}$ | If the intruder is distant and much slower than the ownship, the COC score is never the maximum. | all |
| $P_{3}$ | If the intruder is directly ahead and moving towards the ownship, COC is not the minimum. | all |
| $P_{4}$ | If the intruder is directly ahead and moving away from the ownship, but more slowly, COC is not the minimum. | all |
| $P_{5}$ | If the intruder is near and approaching from the left, strong right is advised. | all |
| $P_{6}$ | If the intruder is far enough away, COC is advised. | all |
| $P_{7}$ | If vertical separation is large, a strong turn is never advised. | `tau = 9` |
| $P_{8}$ | With large vertical separation and previous advisory WL, the network advises COC or WL. | `pra = 2`, `tau = 9` |
| $P_{9}$ | Even if the previous advisory was WR, a nearby intruder makes the network advise SL. | `pra = 3` |
| $P_{10}$ | For a far-away intruder, COC is advised. | all |

`properties/` also contains three combined properties (`P12345610.py`, `P1234568910.py` and `P123467810.py`). They are not among the experiments in this repository, and `ACASXuCertifier.py` accepts only properties 1–10.

## Requirements

- Python 3 with `torch`, `numpy`, `pandas`, `matplotlib` and `onnx`
- [`maraboupy`](https://github.com/NeuralNetworkVerification/Marabou) for formal verification
- [`gurobipy`](https://www.gurobi.com/) with a valid Gurobi licence, for repair and for `realizability-analysis.py`
- `onnxruntime`, only to regenerate the datasets with `datasets.py`
- A LaTeX installation, for `hystogram.py` (it sets `text.usetex = True`)

For example:

```bash
pip install torch numpy pandas matplotlib onnx onnxruntime maraboupy gurobipy
```

Every property module imports Marabou and Gurobi, so both must be installed even to run training alone. A CUDA GPU is used automatically if one is available.

Run every script from the repository root, because the scripts use relative paths.

## Running a single experiment

```bash
python3 ACASXuCertifier.py 1 --pra COC --tau 0 \
    --epsilon 0.0001 --delta 0.001 \
    --learning-rate 0.001 --epochs 1000 --early-stopping 50 \
    --hidden-layers 6 --neurons-per-hidden-layer 50 \
    --counterexamples 50 --sampling 20 --seed 100 \
    --soft-acceleration
```

This example uses the settings from the paper. Run `python3 ACASXuCertifier.py -h` for the full list of options:

| Option | Default | Description |
|---|---|---|
| `property` (positional) | — | Property ID: `1`–`10` |
| `--pra` | `COC` | Previous advisory, by name: `COC`, `WL`, `WR`, `SL` or `SR` (selects the dataset) |
| `--tau` | `0` | Time to loss of vertical separation, in seconds: `0`, `1`, `5`, `10`, `20`, `40`, `60`, `80` or `100` (selects the dataset) |
| `--epsilon` | `0.0001` | ε, the counterexample tolerance |
| `--delta` | `0.01` | Δ, the postcondition margin |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--epochs` | `100` | Maximum epochs per training phase |
| `--early-stopping` | `100` | Patience, in epochs, on the validation loss |
| `--hidden-layers` | `7` | Number of hidden layers |
| `--neurons-per-hidden-layer` | `8` | Width of each hidden layer |
| `--counterexamples` | `1` | Maximum counterexamples repaired per iteration |
| `--sampling` | `20` | Noisy samples generated around each repaired counterexample |
| `--seed` | `1` | Seed for shuffling and splitting the dataset |
| `--soft-acceleration` | off | Add the soft property loss during training |
| `--keep` | off | Keep training the same network across iterations instead of starting fresh |
| `--retain` | off | Accepted but not used |
| `--cores` | `4` | Number of Marabou workers |
| `--force` | off | Overwrite an existing result directory |

Several defaults differ from the paper's settings, so pass the options shown in the example above to reproduce a run from the paper.

### Output

Each run writes to:

```
experiments/<pra>-<tau>-<property>-<epsilon>-<delta>-<lr>-<layers>x<neurons>-<counterexamples>-<sampling>-<seed>-<L|K>[-SA]/
```

Here `<pra>` and `<tau>` are the dataset indices (1–5 and 1–9) from the [Dataset](#dataset) section, not the names passed on the command line. For example, `--pra WL --tau 100` writes to a directory starting with `2-9-`. `L` means a new network was trained at each iteration and `K` means the same network was kept (`--keep`). `-SA` marks soft acceleration. The directory contains:

- `network-<i>.pt`: the PyTorch `state_dict` of the network after CEGIS iteration `i`. The highest `i` is the certified network.
- `stats.json`: per-iteration lists `TrainingTIME`, `OpportunisticSeachTIME`, `FormalVerificationTIME`, `RepairTIME`, `SamplingTIME`, `LossOnValidation`, `LossOnTest`, `OpportunisticCEX`, `FormalVerificationCEX` and `NewPairs`. It also holds `CEGISiterations`, `OriginalInputs` and `TotalTIME` (in seconds). `TotalTIME` is `null` if the run did not finish.

If the result directory already exists, the run stops unless you pass `--force`.

## Reproducing the paper

### All experiments

```bash
python3 Run.py
```

`Run.py` runs, one after another, every combination of property and dataset, with ε = 10⁻⁴, Δ = 10⁻³ and soft acceleration on. It skips properties on datasets where they do not apply. Each run has a 5-hour timeout (`TIMEOUT` in `Run.py`).

> **Note:** as shipped, `Run.py` runs only $P_{1}$–$P_{5}$ (`P = [1,2,3,4,5]`). Set `P = [1,2,3,4,5,6,7,8,9,10]` to run the full grid of the paper, which comes to 330 runs. `experiments.py` defines the same grid.

> **Note:** `experiments/` already contains the paper's results, and `ACASXuCertifier.py` stops on runs whose directory exists. To recompute the results, move `experiments/` somewhere else first. In the shipped results, 12 of the 330 runs did not finish within the timeout.

### Tables and figures

These scripts read `experiments/` and write to the current directory or `img/`:

| Script | Output |
|---|---|
| `python3 experiments.py` | Prints how many runs are present, which are missing and which timed out |
| `python3 latex-baseline.py` | `table-baseline.tex`: the LaTeX table of per-property timeouts and averages (± std) of iterations, times, losses, counterexamples, formal verification calls and new pairs |
| `python3 hystogram.py` | `img/fv-vs-rep-ACASXU-VNNCOMP.pdf`: average formal verification time compared with repair time, per property |
| `python3 convergence.py <k>` | `img/convergence-ACAS-baseline-P<k>.pdf`: the average number of counterexamples during the CEGIS iterations, for property `k` |
| `python3 realizability-analysis.py` | Uses Gurobi to check every pair and triple of $P_{1}$–$P_{10}$ for conflicting postconditions on overlapping preconditions |

> **Note:** `convergence.py` still uses the result layout and parameter grid of the HorizontalCAS experiments (`pra<k>-tau<t>/` subdirectories, Δ = 10⁻², three seeds). As shipped, it finds none of the runs in `experiments/`, so it must be adapted before it can rebuild the `img/convergence-ACAS-*` figures.

## Repository layout

```
ACASXuCertifier.py        CEGIS loop (entry point for one experiment)
training.py               Training loop and asymmetric MSE loss
fm.py                     Selects the property module for each step
conf.py                   Input/output indices and constants
normalization.py          Input bounds and normalisation constants
properties/P*.py          One module per property (and per combination)
datasets.py               Builds datasets/ from the VNN-COMP networks
experiments.py            Experiment grid and result loader
Run.py                    Runs the experiment grid
latex-baseline.py, hystogram.py,
convergence.py            Scripts that build the tables and figures
realizability-analysis.py Checks whether combinations of properties can be satisfied together
datasets/                 ACAS Xu training data (45 .csv files, about 90 MB)
experiments/              Raw results of the paper (about 120 MB)
img/                      Figures used in the paper
```
