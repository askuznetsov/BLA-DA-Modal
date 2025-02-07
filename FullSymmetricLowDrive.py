import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

class NeuralModel:
    def __init__(self):
        # Neuron population parameters
        self.DrF = 0.2  # Baseline drive for F neurons
        self.DrS = 0.2  # Baseline drive for S neurons
        self.ExF = 10    # Excitation parameter for F neurons
        self.ExS = 10    # Excitation parameter for S neurons
        self.TauSF = 3.0      # Time constant for S and F neuron dynamics
        
        # Nonesential connections
        self.W_SS = 0.0
        self.W_FF = 0.0
        self.W_IF = -0.2
        self.W_IS = -0.2
        self.W_ISIF =-0.2
        self.W_IFIS =-0.2
        self.W_IFF = -0.1
        self.W_ISS = -0.1
        
        # IS interneuron parameters
        self.W_ISF = -1.1     # Weight of IS to F neurons
        self.ExIS = 2          # Excitation of IS neurons
        self.DrIS = -0.2       # Baseline drive for IS neurons
        self.W_SIS = .9        # Weight of S to IS neurons

        # IF interneuron parameters
        self.W_IFS = -1.1     # Weight of IF to S neurons
        self.ExIF = 2          # Excitation of IF neurons
        self.DrIF = -0.2       # Baseline drive for IF neurons
        self.W_FIF = .9         # Weight of F to IF neurons

        # Noise level
        self.noise_level = 0.16

    def compute_derivatives(self, F, S, IS, IF, DrF, DrS, dt):
        """Compute derivatives for F, S, IS, and IF neurons."""
        # IS interneuron dynamics
        dISdt = (sigmoid(self.ExIS * (self.W_SIS * S + self.W_IS * IS + self.W_IFIS * IF + self.DrIS)) - IS + self.noise_level * np.random.randn()*np.sqrt(dt))

        # IF interneuron dynamics
        dIFdt = (sigmoid(self.ExIF * (self.W_FIF * F + self.W_IF * IF + self.W_ISIF * IS + self.DrIF)) - IF + self.noise_level * np.random.randn()*np.sqrt(dt))

        # F neuron dynamics
        dFdt = (sigmoid(self.ExF * (self.W_ISF * IS + self.W_FF * F + self.W_IFF * IF + DrF)) - F + self.noise_level * np.random.randn()*np.sqrt(dt)) / self.TauSF

        # S neuron dynamics
        dSdt = (sigmoid(self.ExS * (self.W_IFS * IF + self.W_SS * S + self.W_ISS * IS + DrS)) - S + self.noise_level * np.random.randn()*np.sqrt(dt)) / self.TauSF

        return dFdt, dSdt, dISdt, dIFdt

    def simulate(self, t_max=5000, dt=0.1):
        """Simulate the neural model over time."""
        # Time array
        t = np.linspace(0, t_max, int(t_max / dt) + 1)

        # Initial conditions
        F = np.zeros_like(t)
        S = np.zeros_like(t)
        IS = np.zeros_like(t)
        IF = np.zeros_like(t)
        
        F[0] = 0.6
        S[0] = 0.7
        IS[0] = sigmoid(self.ExIS * (self.W_SIS * S[0] + self.DrIS))
        IF[0] = sigmoid(self.ExIF * (self.W_FIF * F[0] + self.DrIF))

        # Simulation loop
        for i in range(1, len(t)):
            # Define stimulation protocol
            if 0 < t[i] < 1000:  # Low DA state
                DrS = DrF = 0.2
                self.ExF = self.ExS = 5
                self.W_ISF = -1.2
                self.W_IFS = -1.2
                self.TauSF = 1.0
                

            else:  # Hi DA state
                DrS = DrF = 0.2
                self.ExF, self.ExS = 15, 15
                self.W_ISF = -0.8
                self.W_IFS = -.8
                self.TauSF = 3.0
                self.ExIS = self.ExIF = 3
                self.DrIS = self.DrIF = -0.1
                
            # Compute derivatives
            dF, dS, dIS, dIF = self.compute_derivatives(F[i-1], S[i-1], IS[i-1], IF[i-1], DrF, DrS,dt)

            # Update state variables
            F[i] = F[i-1] + dF * dt
            S[i] = S[i-1] + dS * dt
            IS[i] = IS[i-1] + dIS * dt
            IF[i] = IF[i-1] + dIF * dt

        return t, F, S, IS, IF

    def plot_results(self, t, F, S, IS, IF):
        """Plot simulation results."""
        plt.figure(figsize=(5, 3))
        plt.plot(t, S, label="S", color='blue')
        plt.plot(t, F, label="F", color='red')
        #plt.plot(t, IS, label="IS", color='green')
        #plt.plot(t, IF, label="IF", color='purple')
        plt.xlabel("Time")
        plt.ylabel("Activity")

        # Highlight different states
        for i in range(1, 2):
            plt.axvspan(i*1000, 1000+i*1000, facecolor='0.2', alpha=0.2)

        plt.xlim(0, 2000)
        plt.ylim(0, 1)
        plt.text(500, 0.6, 'Low DA', va='bottom', ha='center', rotation=0)
        plt.text(1500, 0.6, 'High DA', va='bottom', ha='center', rotation=0)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('Figure2A.pdf')
        plt.show()

def main():
    model = NeuralModel()
    t, F, S, IS, IF = model.simulate()
    model.plot_results(t, F, S, IS, IF)

if __name__ == "__main__":
    main()
