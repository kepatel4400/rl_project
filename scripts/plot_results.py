import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def plot_step_vs_value(csv_paths, output_path='step_vs_value_plot.png'):
    """
    Plots Step vs Value for multiple CSV files.
    
    Args:
        csv_paths: List of paths to CSV files
        output_path: Path to save the output image
    """
    plt.figure(figsize=(12, 7))
    
    for path in csv_paths:
        try:
            # Read CSV file
            df = pd.read_csv(path)
            
            # Extract Step and Value columns
            steps = df['Step']
            values = df['Value']
            
            # Use filename without extension as label
            label = os.path.splitext(os.path.basename(path))[0]
            
            # Plot the data
            plt.plot(steps, values, linestyle='-', label=label)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # Add labels and title
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Success rate', fontsize=12)
    plt.title('Step vs Success Rate: Multimodal With vs Without Goal-conditioned', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parent.parent / "results"
    output = Path("outputs/plots/step_vs_success_rate_plot_multimodal_compare.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_step_vs_value(
        [results_dir / "multimodal_no_goal.csv", results_dir / "multimodal.csv"],
        output_path=output,
    )
