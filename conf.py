InputSIZE = 5
OutputSIZE = 5

# inputs
rho    = 0
theta  = 1
psi    = 2
vown   = 3
vint   = 4

# outputs
coc = 0
wl  = 1
wr  = 2
sl  = 3
sr  = 4

#print(norm_input_min)
#print(norm_input_max)

ActionsDifferentFromCOC = [wl, wr, sl, sr]
ActionsDifferentFromSR  = [coc, wl, wr, sl]
ActionsDifferentFromSL  = [coc, wl, wr, sr]
ActionsDifferentFromSL_SR = [coc, wl, wr]
ActionsDifferentFromCOC_WL = [wr, sl, sr]

AMPLIFIER = 40