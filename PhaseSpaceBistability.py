import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return 1 / (1 + np.exp(-x))

# Define parameters
DrF = 0.4 # defines the max and min activation of the F neurons, but not the transition
DrS = 0.4 # was 0.5
ExF=15 # changes the min and max activation of the F neurons too much
ExS=15

# Parameters of the PV neurons
W_PVF=-1 # greater value allows for greater inhibition of F cells
ExPV=3 # 3 steepens the F nullcline
DrPV=-0.2 # -0.2 shifts the transition
W_SPV=1 # shifts the transition

# Parameters of the CCK neurons
W_CCKS=-1 # greater value allows for greater inhibition of S cells
ExCCK=3 # steepens the S nullcline
DrCCK=-0.2 # shifts the transition
W_FCCK=1 # shifts the transition

# Hi DA state
#DrS = 0.72 # was 0.75
ExF=15 # was 10, 8 works
ExS=15
# Parameters of the PV neurons
W_PVF=-0.8 # was 1; 0.8 works
#ExPV=3.3 # was 3; 4 works
#DrPV=-0.16 # was -0.2; -0.15 works
# Parameters of the CCK neurons
W_CCKS=-0.8 # was 1; 0.8 works

def dFdt(F, S, DrF, DrS):
    PV=f(ExPV*(W_SPV * S + DrPV))
    return f(ExF*(W_PVF * PV + DrF)) - F
  #return f(W * S + DrF) - F

def dSdt(F, S, DrF, DrS):
    CCK=f(ExCCK*(W_FCCK * F + DrCCK))
    return f(ExS*(W_CCKS * CCK + DrS)) - S
  #return f(ExS*(W * F + DrS)) - S


# Define initial conditions
F_init = 0.6
S_init = 0.7

# Define time range
t_max = 100
dt = 0.1
t = np.linspace(0, t_max, int(t_max / dt) + 1)

# Solve the system of differential equations
F = np.zeros_like(t)
S = np.zeros_like(t)
F[0] = F_init
S[0] = S_init
for i in range(1, len(t)):
  F[i] = F[i-1] + dFdt(F[i-1], S[i-1], DrF, DrS) * dt
  S[i] = S[i-1] + dSdt(F[i-1], S[i-1], DrF, DrS) * dt

# Calculate nullclines
S_nullcline_F = np.linspace(0, 1, 100)
S_nullcline_S = dSdt(S_nullcline_F, 0, DrF, DrS)
F_nullcline_S = np.linspace(0, 1, 100)
F_nullcline_F = dFdt(0, F_nullcline_S, DrF, DrS)

# Plot the trajectory and nullclines
plt.figure(figsize=(5.5, 5))
plt.plot(F, S, label="Trajectory", color="black")
plt.plot(S_nullcline_F, S_nullcline_S, label="dS/dt=0", color="blue")
plt.plot(F_nullcline_F, F_nullcline_S, label="dF/dt=0", color="red")
#plt.plot(S_nc_F_HDA, S_nc_S_HDA, label="dS/dt=0, High DA", color="blue")
#plt.plot(F_nc_F_HDA, F_nc_S_HDA, label="dF/dt=0, High DA", color="red")
#plt.plot(F_HDA, S_HDA, label="DA increase", color="black")
#plt.plot(S_nc_F_LDA, S_nc_S_LDA, label="dS/dt=0, Low DA", linestyle='--',color="blue")
#plt.plot(F_nc_F_LDA, F_nc_S_LDA, label="dF/dt=0, Low DA", linestyle='--',color="red")
#plt.plot(F_LDA, S_LDA, label="Low DA")

# Add arrow using plt.arrow
ap = 3
plt.arrow(F[ap], S[ap], F[ap+1]-F[ap], S[ap+1]-S[ap], head_width=0.03, head_length=0.06, color='black')
lp = 0.951
arrow2=plt.arrow(0.72, lp, 0.01, 0, head_width=0.03, head_length=0.06, color='black')
arrow2.set_zorder(10)  # Adjust z-order as needed (higher is more front)

plt.xlabel("F")
plt.ylabel("S")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.savefig('Figure3C.pdf')
plt.show()
