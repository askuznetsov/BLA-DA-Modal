import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return 1 / (1 + np.exp(-x))

# Define parameters
DrF = 0.5 # defines the max and min activation of the F neurons, but not the transition
DrS = 0.5 # was 0.5
ExF=10 # changes the min and max activation of the F neurons too much
ExS=10

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
#S_nullcline_2 = f(W * F_nullcline + DrS)
#PV_nul = PV=f(ExPV*(W_SPV * F_nullcline_S + DrPV))

# Low drive state
DrS = 0.2 # was 0.5, 0.35 works
DrF = 0.2 # was 0.5# Re-calculate nullclines

# Low DA state
DrS = 0.25 # was 0.5, 0.55 works
ExF=5 # was 10, 8 works
ExS=5
# Parameters of the PV neurons
W_PVF=-1.2 # was 1
ExPV=2.5 # was 3; 2 works
DrPV=-0.25 # was -0.2
# Parameters of the CCK neurons
W_CCKS=-1.2 # was 1

# Re-calculate nullclines
S_nc_F_LDA = np.linspace(0, 1, 100)
S_nc_S_LDA = dSdt(S_nc_F_LDA, 0, DrF, DrS)
F_nc_S_LDA = np.linspace(0, 1, 100)
F_nc_F_LDA = dFdt(0, F_nc_S_LDA, DrF, DrS)

# Solve the system of differential equations at high DA
F_LDA = np.zeros_like(t)
S_LDA = np.zeros_like(t)
F_LDA[0] = F_init
S_LDA[0] = S_init
for i in range(1, len(t)):
  F_LDA[i] = F_LDA[i-1] + dFdt(F_LDA[i-1], S_LDA[i-1], DrF, DrS) * dt
  S_LDA[i] = S_LDA[i-1] + dSdt(F_LDA[i-1], S_LDA[i-1], DrF, DrS) * dt

# Hi DA state
DrS = 0.16 # was 0.5, 0.55 works
ExF=15 # was 10, 8 works
ExS=15
# Parameters of the PV neurons
W_PVF=-0.8 # was 1
ExPV=3.5 # was 3; 2 works
DrPV=-0.13 # was -0.2
# Parameters of the CCK neurons
W_CCKS=-0.8 # was 1

# Re-calculate nullclines
S_nc_F_HDA = np.linspace(0, 1, 100)
S_nc_S_HDA = dSdt(S_nc_F_HDA, 0, DrF, DrS)
F_nc_S_HDA = np.linspace(0, 1, 100)
F_nc_F_HDA = dFdt(0, F_nc_S_HDA, DrF, DrS)

# Solve the system of differential equations at high DA
F_HDA = np.zeros_like(t)
S_HDA = np.zeros_like(t)
F_HDA[0] = F_LDA[-1]
S_HDA[0] = S_LDA[-1]
for i in range(1, len(t)):
  F_HDA[i] = F_HDA[i-1] + dFdt(F_HDA[i-1], S_HDA[i-1], DrF, DrS) * dt
  S_HDA[i] = S_HDA[i-1] + dSdt(F_HDA[i-1], S_HDA[i-1], DrF, DrS) * dt

# Plot the trajectory and nullclines
plt.figure(figsize=(5.5, 5))
#plt.plot(F, S, label="Trajectory")
#plt.plot(S_nullcline_F, S_nullcline_S, label="dS/dt=0")
#plt.plot(F_nullcline_F, F_nullcline_S, label="dF/dt=0")
plt.plot(S_nc_F_HDA, S_nc_S_HDA, label="dS/dt=0, High DA", color="blue")
plt.plot(F_nc_F_HDA, F_nc_S_HDA, label="dF/dt=0, High DA", color="red")
plt.plot(F_HDA, S_HDA, label="DA Increase", color="black")
plt.plot(S_nc_F_LDA, S_nc_S_LDA, label="dS/dt=0, Low DA", linestyle='--',color="blue")
plt.plot(F_nc_F_LDA, F_nc_S_LDA, label="dF/dt=0, Low DA", linestyle='--',color="red")
#plt.plot(F_LDA, S_LDA, label="Low DA")
# Add arrow using plt.arrow
ap = 2
plt.arrow(F_HDA[ap], S_HDA[ap], F_HDA[ap+1]-F_HDA[ap], S_HDA[ap+1]-S_HDA[ap], head_width=0.03, head_length=0.06, color='black')
lp = 0.845
arrow2=plt.arrow(0.59, lp, 0.01, 0, head_width=0.03, head_length=0.06, color='black')
arrow2.set_zorder(10)  # Adjust z-order as needed (higher is more front)

plt.xlabel("F")
plt.ylabel("S")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.savefig('Figure2C.pdf')
plt.show()
