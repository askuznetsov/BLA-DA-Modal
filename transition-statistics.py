import numpy as np
import matplotlib.pyplot as plt
from TransStat import NeuralModel

def run_transition_analysis(drive_offsets=np.linspace(-0.1, 0.1, 31), num_trials=30,
                             t_max=1000, dt=0.1):
    """
    Analyze neural model response across different F neuron drive levels.
    
    Parameters:
    - drive_offsets: Array of offsets from base drive (0.4)
    - num_trials: Number of trials for each drive level
    - t_max: Total simulation time
    - dt: Time step
    
    Returns:
    - Dictionary with statistical results for each drive offset
    """
    # Initialize results dictionary
    results = {}
    
    for drive_offset in drive_offsets:
        # Store trial results for this drive offset
        f_averages = []
        s_averages = []
        
        for _ in range(num_trials):
            # Create a new model for each trial to ensure independent runs
            model = NeuralModel()
            
            # Modify simulation to use variable drive
            def modified_simulate():
                t, F, S, IS, IF = model.simulate(t_max, dt)
                
                # Compute average for last half of simulation 
                # during the neutral/constant state
                start_index = int(len(t) * 0.25)
                f_avg = np.mean(F[start_index:])
                s_avg = np.mean(S[start_index:])
                
                return f_avg, s_avg
            
            # Update base parameters
            model.DrF = 0.4 + drive_offset
            
            # Run trial and store results
            f_avg, s_avg = modified_simulate()
            f_averages.append(f_avg)
            s_averages.append(s_avg)
        
        # Compute statistics for this drive offset
        results[drive_offset] = {
            'f_mean': np.mean(f_averages),
            'f_std': np.std(f_averages),
            's_mean': np.mean(s_averages),
            's_std': np.std(s_averages),
            'difference_mean': np.mean(np.array(f_averages) - np.array(s_averages)),
            'difference_std': np.std(np.array(f_averages) - np.array(s_averages))
        }
    
    return results

def plot_transition_analysis(results):
    """
    Plot results of transition analysis.
    
    Parameters:
    - results: Dictionary of statistical results from run_transition_analysis
    """
    drive_offsets = list(results.keys())
    
    # Prepare data for plotting
    difference_means = [results[offset]['difference_mean'] for offset in drive_offsets]
    difference_stds = [results[offset]['difference_std'] for offset in drive_offsets]
    
    plt.figure(figsize=(6, 3))
    
    # Plot mean difference
    plt.errorbar(drive_offsets, difference_means, 
                 yerr=difference_stds, 
                 fmt='o-', 
                 capsize=5,
                 label='F-S Activity Difference')
    
    plt.xlabel('Drive Offset to F Neurons')
    plt.ylabel('Average Activity Difference')
    #plt.title('Neural Model Transition Analysis')
    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Figure6.pdf')
    plt.show()

def main():
    # Run analysis
    results = run_transition_analysis()
    
    # Print detailed results
    for offset, stats in results.items():
        print(f"\nDrive Offset: {offset}")
        print(f"F Neurons - Mean: {stats['f_mean']:.4f} ± {stats['f_std']:.4f}")
        print(f"S Neurons - Mean: {stats['s_mean']:.4f} ± {stats['s_std']:.4f}")
        print(f"Difference (F-S) - Mean: {stats['difference_mean']:.4f} ± {stats['difference_std']:.4f}")
    
    # Plot results
    plot_transition_analysis(results)

if __name__ == "__main__":
    main()
