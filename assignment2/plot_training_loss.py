import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open("metrics.json") as f:
    metrics = json.load(f)

epochs = [m["epoch"] for m in metrics]
train_loss = [m["train_loss"] for m in metrics]
val_loss = [m["val_loss"] for m in metrics]

best_epoch = min(metrics, key=lambda m: m["val_loss"])["epoch"]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(epochs, train_loss, label="Train Loss", color="#2196F3", linewidth=2, marker="o", markersize=4)
ax.plot(epochs, val_loss, label="Val Loss", color="#F44336", linewidth=2, marker="s", markersize=4)

ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.2, label=f"Best val (epoch {best_epoch})")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Training & Validation Loss", fontsize=13)
ax.legend(fontsize=11)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
print("Saved training_loss.png")
