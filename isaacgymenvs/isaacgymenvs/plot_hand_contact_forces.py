import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from glob import glob

# Set the style for better-looking plots
plt.style.use('seaborn')
sns.set_palette("husl")

def load_latest_log():
    """Load the most recent log file from the logs directory."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')
    log_files = glob(os.path.join(log_dir, 'shadow_hand_contact_forces_*.csv'))
    if not log_files:
        raise FileNotFoundError("No log files found in logs directory")
    latest_log = max(log_files, key=os.path.getctime)
    return pd.read_csv(latest_log)

def plot_contact_forces(df):
    """Create plots for contact forces and torques for each fingertip."""
    # Get unique fingertip names
    fingertips = df['Fingertip'].unique()
    num_fingertips = len(fingertips)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Shadow Hand Contact Forces and Torques Over Time', fontsize=16)
    
    # Create grid of subplots: 2 rows (forces and torques) x num_fingertips columns
    gs = fig.add_gridspec(3, num_fingertips, hspace=0.3, wspace=0.3)
    
    # Plot for each fingertip
    for idx, fingertip in enumerate(fingertips):
        fingertip_data = df[df['Fingertip'] == fingertip]
        
        # Force magnitude plot
        ax_mag = fig.add_subplot(gs[0, idx])
        ax_mag.plot(fingertip_data['Step'], fingertip_data['Force_Magnitude'])
        ax_mag.set_title(f'{fingertip}\nForce Magnitude')
        ax_mag.set_xlabel('Step')
        ax_mag.set_ylabel('Force (N)')
        ax_mag.grid(True)
        
        # Force components plot
        ax_force = fig.add_subplot(gs[1, idx])
        ax_force.plot(fingertip_data['Step'], fingertip_data['Force_X'], label='X')
        ax_force.plot(fingertip_data['Step'], fingertip_data['Force_Y'], label='Y')
        ax_force.plot(fingertip_data['Step'], fingertip_data['Force_Z'], label='Z')
        ax_force.set_title('Force Components')
        ax_force.set_xlabel('Step')
        ax_force.set_ylabel('Force (N)')
        ax_force.legend()
        ax_force.grid(True)
        
        # Torque components plot
        ax_torque = fig.add_subplot(gs[2, idx])
        ax_torque.plot(fingertip_data['Step'], fingertip_data['Torque_X'], label='X')
        ax_torque.plot(fingertip_data['Step'], fingertip_data['Torque_Y'], label='Y')
        ax_torque.plot(fingertip_data['Step'], fingertip_data['Torque_Z'], label='Z')
        ax_torque.set_title('Torque Components')
        ax_torque.set_xlabel('Step')
        ax_torque.set_ylabel('Torque (Nm)')
        ax_torque.legend()
        ax_torque.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    plt.savefig(os.path.join(output_dir, 'hand_contact_forces_plot.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def analyze_data(df):
    """Print statistical analysis of the contact forces."""
    print("\nContact Force Analysis:")
    print("-" * 50)
    
    # Group by fingertip and calculate statistics
    stats = df.groupby('Fingertip').agg({
        'Force_Magnitude': ['mean', 'std', 'max'],
        'Force_Z': ['mean', 'std', 'max'],
        'Torque_X': ['mean', 'std', 'max'],
        'Torque_Y': ['mean', 'std', 'max'],
        'Torque_Z': ['mean', 'std', 'max']
    }).round(3)
    
    print("\nForce and Torque Statistics by Fingertip:")
    print(stats)
    
    # Calculate total force distribution across fingertips
    total_force = df.groupby('Fingertip')['Force_Magnitude'].mean()
    print("\nAverage Load Distribution (% of total):")
    print((total_force / total_force.sum() * 100).round(2))

def main():
    try:
        # Load data
        df = load_latest_log()
        print(f"Loaded data with {len(df)} records")
        
        # Create plots
        print("Creating plots...")
        plot_contact_forces(df)
        print(f"Plots saved to {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'hand_contact_forces_plot.png')}")
        
        # Analyze data
        analyze_data(df)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 