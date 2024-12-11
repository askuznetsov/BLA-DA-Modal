import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

class RandomNeuralPopulationModel:
    def __init__(self, num_neurons_per_group=100):
        #np.random.seed(42)
        self.num_neurons = num_neurons_per_group

        # Connectivity parameters
        # Within groups
        self.W_E1E1 = 0.10  # Within-group S excitation
        self.W_E2E2 = 0.10  # Within-group F excitation
        self.W_I1I1 = -0.20  # 0.6 Within-group IS excitation
        self.W_I2I2 = -0.20  # 0.6 Within-group IF excitation

        # Between-group connectivity
        self.W_E1I1 = .90  # 1 S to IS
        self.W_I1E2 = -1.1  # IS to F
        self.W_E2I2 = .90  # 1 F to IF
        self.W_I2E1 = -1.1  # IF to S
        
        self.W_I1I2 = -.2  # IS to IF
        self.W_I2I1 = -.2  # IF to IS
        self.W_I1E1 = -.1  # IS to S
        self.W_I2E2 = -.1  # IF to F

        # Baseline drives and time constants
        self.DriveS = 0.3 # 0.4
        self.DriveF = 0.3
        self.DriveI1 = -0.20 #0
        self.DriveI2 = -0.20 #0
        self.ExcI = 3.0
        self.ExcE = 7.0  # Low excitability at low DA
        self.tauE = 3.0
        self.tauI = 1.0  # tau for inhibitory neurons
        self.noise_level = 0.8
        
        # Within-group connectivity matrices
        self.M_E1E1 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        self.M_E2E2 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        self.M_I1I1 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        self.M_I2I2 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        
        # Between-group connectivity matrices
        self.M_I1I2 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        self.M_I2I1 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        
        self.M_E1I1 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        self.M_I1E2 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        self.M_E2I2 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        self.M_I2E1 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        self.M_I1E1 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        self.M_I2E2 = np.random.uniform(0.7, 1.3, (num_neurons_per_group, num_neurons_per_group))
        
        self.V_DS = np.random.uniform(0.95, 1.05, (num_neurons_per_group))
        self.V_DS[0] = 0.
        
    def compute_neuron_derivative(self, neuron_value, external_input):
        """Compute derivative for a single neuron with noise."""
        derivative = (sigmoid(external_input) - neuron_value) + \
                    self.noise_level * np.random.randn()
        return derivative

    def simulate(self, t_max=5000, dt=0.1):
        """Simulate the neural population model."""
        t = np.linspace(0, t_max, int(t_max / dt) + 1)

        # Initialize neuron populations
        E1 = np.random.rand(self.num_neurons)
        E2 = np.random.rand(self.num_neurons)
        I1 = np.random.rand(self.num_neurons)
        I2 = np.random.rand(self.num_neurons)

        # Arrays to store results
        E1_avg = np.zeros_like(t)
        E2_avg = np.zeros_like(t)
        E1_0 = np.zeros_like(t)  # First neuron in E1
        E1_1 = np.zeros_like(t)  # Second neuron in E1
        E2_0 = np.zeros_like(t)  # First neuron in E2
        E2_1 = np.zeros_like(t)  # Second neuron in E2
        
        E1_avg[0] = np.mean(E1)
        E2_avg[0] = np.mean(E2)
        E1_0[0] = E1[0]
        E1_1[0] = E1[1]
        E2_0[0] = E2[0]
        E2_1[0] = E2[1]

        # Simulation loop
        for i in range(1, len(t)):
            # Define stimulation protocol
            if 0 < t[i] < 200:  # Safety stimulus
                self.DriveS = 0.3
                self.DriveF = 0.4
                self.ExcE = 12
                self.W_I1E2 = -0.8  # I1 to E2
                self.W_I2E1 = -0.8  # I2 to E1
            elif 1000 < t[i] < 1200:  # Fear stimulus
                self.DriveS = 0.4
                self.DriveF = 0.3
                self.ExcE = 12
                self.W_I1E2 = -0.8  # I1 to E2
                self.W_I2E1 = -0.8  # I2 to E1
            elif 2000 < t[i] < 2200 or 3000 < t[i] < 3200 or 4000 < t[i] < 4200:  # Neutral stimulus
                self.DriveS = self.DriveF = 0.3
                self.ExcE = 12
                self.W_I1E2 = -0.8  # I1 to E2
                self.W_I2E1 = -0.8  # I2 to E1
            else:  # Low DA state
                self.DriveS = self.DriveF = 0.3
                self.ExcE = 7
                self.W_I1E2 = -1.0  # -1 I1 to E2
                self.W_I2E1 = -1.0  # -1 I2 to E1

            # Update each neuron individually
            for j in range(self.num_neurons):
                
                # I1 neurons - IS
                I1_input = self.ExcI * (np.sum(self.W_E1I1 * self.M_E1I1[j] * E1)/self.num_neurons +
                                        np.sum(self.W_I1I1 * self.M_I1I1[j] * I1)/self.num_neurons + np.sum(self.W_I2I1 * self.M_I2I1[j] * I2)/self.num_neurons + self.DriveI1)
                I1[j] += self.compute_neuron_derivative(I1[j], I1_input) * dt / self.tauI


                # I2 neurons - IF
                I2_input = self.ExcI * (np.sum(self.W_E2I2 * self.M_E2I2[j] * E2)/self.num_neurons +
                                        np.sum(self.W_I2I2 * self.M_I2I2[j] * I2)/self.num_neurons + np.sum(self.W_I1I2 * self.M_I1I2[j] * I1)/self.num_neurons + self.DriveI2)
                I2[j] += self.compute_neuron_derivative(I2[j], I2_input) * dt / self.tauI
                
                # E2 neurons - Fear
                E2_input = self.ExcE * (np.sum(self.W_E2E2 * self.M_E2E2[j] * E2)/self.num_neurons +
                                        np.sum(self.W_I1E2 * self.M_I1E2[j] * I1)/self.num_neurons + np.sum(self.W_I2E2 * self.M_I2E2[j] * I2)/self.num_neurons +
                                        self.DriveF)
                E2[j] += self.compute_neuron_derivative(E2[j], E2_input) * dt / self.tauE
                
                # E1 neurons - Safety
                E1_input = self.ExcE * (np.sum(self.W_E1E1 * self.M_E1E1[j] * E1)/self.num_neurons +
                                        np.sum(self.W_I2E1 * self.M_I2E1[j] * I2)/self.num_neurons + np.sum(self.W_I1E1 * self.M_I1E1[j] * I1)/self.num_neurons +
                                        self.DriveS*self.V_DS[j])
                E1[j] += self.compute_neuron_derivative(E1[j], E1_input) * dt / self.tauE

            # Store average activities
            E1_avg[i] = np.mean(E1)
            E2_avg[i] = np.mean(E2)
            E1_0[i] = E1[0]
            E1_1[i] = E1[1]
            E2_0[i] = E2[0]
            E2_1[i] = E2[1]

        return t, E1_avg, E2_avg, E1_0, E1_1, E2_0, E2_1

    def plot_results(self, t, E1_avg, E2_avg, E1_0, E1_1, E2_0, E2_1):
        """Plot simulation results."""
        plt.figure(figsize=(10, 3))
        plt.plot(t, E1_avg, label="S Average", color='blue', linewidth=2)
        plt.plot(t, E1_0, label="S Neuron 1", color='blue', linestyle='--', alpha=0.5)
        #plt.plot(t, E1_1, label="S Neuron 2", color='blue', linestyle=':', alpha=0.5)
        
        plt.plot(t, E2_avg, label="F Average", color='red', linewidth=2)
        #plt.plot(t, E2_0, label="F Neuron 1", color='red', linestyle='--', alpha=0.5)
        #plt.plot(t, E2_1, label="F Neuron 2", color='red', linestyle=':', alpha=0.5)
        
        plt.xlabel("Time")
        plt.ylabel("Average Neuron Activity")

        # Highlight different states
        for i in range(0, 5):
            plt.axvspan(i*1000, 200+i*1000, facecolor='0.2', alpha=0.2)

        plt.xlim(0, 5000)
        plt.ylim(0, 1)

        # State labels
        states = ['Negative', 'Positive', 'Neutral', 'Neutral', 'Neutral']
        for i, state in enumerate(states):
            plt.text(100 + i*1000, 0.4, state, va='bottom', ha='center', rotation=90)

        plt.legend()
        plt.tight_layout()
        plt.savefig('FigureSI4.pdf')
        plt.show()

def main():
    model = RandomNeuralPopulationModel(num_neurons_per_group=100)
    t, E1_avg, E2_avg, E1_0, E1_1, E2_0, E2_1 = model.simulate()
    model.plot_results(t, E1_avg, E2_avg, E1_0, E1_1, E2_0, E2_1)

if __name__ == "__main__":
    main()
