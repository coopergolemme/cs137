import subprocess
import json
import matplotlib.pyplot as plt
import os

def run_experiment():
    epochs = 15
    batch_size = 16
    
    # 1. Run Baseline (no residuals)
    print("=== Running Baseline Model ===")
    subprocess.run([
        "python", "train.py", 
        "--epochs", str(epochs), 
        "--batch_size", str(batch_size), 
        "--metrics_file", "metrics_baseline.json",
        "--save_dir", "baseline_output"
    ], check=True)

    # 2. Run Residual Model
    print("=== Running Residual Model ===")
    subprocess.run([
        "python", "train.py", 
        "--use_residual",
        "--epochs", str(epochs), 
        "--batch_size", str(batch_size), 
        "--metrics_file", "metrics_residual.json",
        "--save_dir", "residual_output"
    ], check=True)

def plot_metrics():
    print("=== Formatting Plots ===")
    
    with open("metrics_baseline.json", "r") as f:
        baseline_metrics = json.load(f)
        
    with open("metrics_residual.json", "r") as f:
        residual_metrics = json.load(f)
        
    epochs_b = [m["epoch"] for m in baseline_metrics]
    train_loss_b = [m["train_loss"] for m in baseline_metrics]
    val_auc_b = [m["val_auc"] for m in baseline_metrics]
    val_rmse_b = [m["val_rmse_base"] for m in baseline_metrics]

    epochs_r = [m["epoch"] for m in residual_metrics]
    train_loss_r = [m["train_loss"] for m in residual_metrics]
    val_auc_r = [m["val_auc"] for m in residual_metrics]
    val_rmse_r = [m["val_rmse_base"] for m in residual_metrics]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Train Loss
    axs[0].plot(epochs_b, train_loss_b, label="Baseline", marker="o")
    axs[0].plot(epochs_r, train_loss_r, label="Residuals", marker="x")
    axs[0].set_title("Training Loss vs. Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Validation AUC
    axs[1].plot(epochs_b, val_auc_b, label="Baseline", marker="o")
    axs[1].plot(epochs_r, val_auc_r, label="Residuals", marker="x")
    axs[1].set_title("Validation AUC vs. Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("AUC")
    axs[1].legend()
    axs[1].grid(True)

    # Validation RMSE
    axs[2].plot(epochs_b, val_rmse_b, label="Baseline", marker="o")
    axs[2].plot(epochs_r, val_rmse_r, label="Residuals", marker="x")
    axs[2].set_title("Validation Base RMSE vs. Epochs")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("RMSE")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("residual_study_results.png", dpi=300)
    print("Saved plot to residual_study_results.png")

if __name__ == "__main__":
    os.makedirs("baseline_output", exist_ok=True)
    os.makedirs("residual_output", exist_ok=True)
    
    # Run the models (comment out if you already have the json files and just want to re-plot)
    run_experiment()
    
    # Generate the plots
    plot_metrics()
