import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style for better-looking plots
plt.style.use('seaborn')
sns.set_palette("husl")

# Specific paths
CSV_PATH = "/scratch/bdes/haorany7/Eureka-Research/isaacgymenvs/logs/ant_contact_forces_20250516_150141.csv"
OUTPUT_DIR = "/scratch/bdes/haorany7/Eureka-Research/isaacgymenvs/logs"

def plot_contact_forces(df):
    """Create plots for contact forces and torques."""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Ant Contact Forces and Torques Over Time', fontsize=16)

    # Get unique foot names
    feet = df['Foot'].unique()
    
    # Plot Force Magnitude
    for foot in feet:
        foot_data = df[df['Foot'] == foot]
        ax1.plot(foot_data['Step'], foot_data['Force_Magnitude'], label=foot)
    ax1.set_title('Force Magnitude')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Force (N)')
    ax1.legend()
    ax1.grid(True)

    # Plot Force Components
    for foot in feet:
        foot_data = df[df['Foot'] == foot]
        ax2.plot(foot_data['Step'], foot_data['Force_Z'], label=f'{foot} (Z)')
    ax2.set_title('Vertical Force (Z)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Force (N)')
    ax2.legend()
    ax2.grid(True)

    # Plot Horizontal Forces
    for foot in feet:
        foot_data = df[df['Foot'] == foot]
        ax3.plot(foot_data['Step'], foot_data['Force_X'], label=f'{foot} (X)')
        ax3.plot(foot_data['Step'], foot_data['Force_Y'], label=f'{foot} (Y)', linestyle='--')
    ax3.set_title('Horizontal Forces (X, Y)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Force (N)')
    ax3.legend()
    ax3.grid(True)

    # Plot Torques
    for foot in feet:
        foot_data = df[df['Foot'] == foot]
        for component, style in zip(['Torque_X', 'Torque_Y', 'Torque_Z'], ['-', '--', ':']):
            ax4.plot(foot_data['Step'], foot_data[component], 
                    label=f'{foot} ({component[-1]})', linestyle=style)
    ax4.set_title('Torques')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Torque (Nm)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'contact_forces_plot.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {output_path}")
    plt.close()

def analyze_data(df):
    """Print statistical analysis of the contact forces."""
    print("\nContact Force Analysis:")
    print("-" * 50)
    
    # Group by foot and calculate statistics
    stats = df.groupby('Foot').agg({
        'Force_Magnitude': ['mean', 'std', 'max'],
        'Force_Z': ['mean', 'std', 'max'],
        'Torque_X': ['mean', 'std', 'max'],
        'Torque_Y': ['mean', 'std', 'max'],
        'Torque_Z': ['mean', 'std', 'max']
    }).round(3)
    
    print("\nForce and Torque Statistics by Foot:")
    print(stats)
    
    # Calculate total load distribution
    total_force = df.groupby('Foot')['Force_Magnitude'].mean()
    print("\nAverage Load Distribution (% of total):")
    print((total_force / total_force.sum() * 100).round(2))

def main():
    try:
        # Load data
        print(f"Loading data from: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        
        # Create plots
        print("Creating plots...")
        plot_contact_forces(df)
        
        # Analyze data
        analyze_data(df)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 