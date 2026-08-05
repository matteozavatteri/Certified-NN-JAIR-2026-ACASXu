import conf

# Min and max values used to bound the inputs
raw_input_min  = [0, -3.141593, -3.141593, 100, 0]
raw_input_max = [60760, 3.141593, 3.141593, 1200, 1200]

# Mean and range values for normalizing the inputs and outputs
raw_input_mean  = [1.9791091e+04, 0, 0, 650, 600]
raw_input_range = [60261, 6.28318530718, 6.28318530718, 1100, 1200]

raw_output_mean  = 7.5188840201005975
raw_output_range = 373.94992


def normalize_rho(raw):
	return (raw - raw_input_mean[conf.rho]) / raw_input_range[conf.rho]

def normalize_theta(raw):
	return (raw - raw_input_mean[conf.theta]) / raw_input_range[conf.theta]

def normalize_psi(raw):
	return (raw - raw_input_mean[conf.psi]) / raw_input_range[conf.psi]

def normalize_vown(raw):
	return (raw - raw_input_mean[conf.vown]) / raw_input_range[conf.vown]

def normalize_vint(raw):
	return (raw - raw_input_mean[conf.vint]) / raw_input_range[conf.vint]

def normalize_output(raw):
	return (raw - raw_output_mean) / raw_output_range


# input bounds (normalized)
Input_LB = [normalize_rho(raw_input_min[conf.rho]),
			normalize_theta(raw_input_min[conf.theta]),
			normalize_psi(raw_input_min[conf.psi]),
			normalize_vown(raw_input_min[conf.vown]),
			normalize_vint(raw_input_min[conf.vint])]

Input_UB = [normalize_rho(raw_input_max[conf.rho]),
			normalize_theta(raw_input_max[conf.theta]),
			normalize_psi(raw_input_max[conf.psi]),
			normalize_vown(raw_input_max[conf.vown]),
			normalize_vint(raw_input_max[conf.vint])]