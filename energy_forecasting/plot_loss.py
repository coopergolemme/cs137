import re
import matplotlib.pyplot as plt
import argparse
import os

def parse_and_plot(log_file):
    step_losses = []
    epoch_train_losses = []
    epoch_val_losses = []
    epochs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match step loss: "  Epoch 1 [436/2181]  loss: 102.87 MW"
            step_match = re.search(r'Epoch\s+\d+\s+\[\d+/\d+\]\s+loss:\s+([0-9.]+)', line)
            if step_match:
                step_losses.append(float(step_match.group(1)))
            
            # Match epoch loss: "Epoch   1/50  train: 104.35  val: 90.56"
            epoch_match = re.search(r'Epoch\s+(\d+)/\d+\s+train:\s+([0-9.]+)\s+val:\s+([0-9.]+)', line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
                epoch_train_losses.append(float(epoch_match.group(2)))
                epoch_val_losses.append(float(epoch_match.group(3)))

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Step loss
    ax1.plot(step_losses, alpha=0.6, color='blue', label='Step Loss')
    ax1.set_title('Training Loss per Step')
    ax1.set_xlabel('Steps (reported intervals)')
    ax1.set_ylabel('Loss (MW)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Epoch train and val loss
    ax2.plot(epochs, epoch_train_losses, marker='o', color='blue', label='Train Loss')
    ax2.plot(epochs, epoch_val_losses, marker='o', color='red', label='Val Loss')
    ax2.set_title('Training and Validation Loss per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MW)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    
    output_path = os.path.splitext(log_file)[0] + '_loss_plot.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', help='Path to the log file')
    args = parser.parse_args()
    parse_and_plot(args.log_file)
