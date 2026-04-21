import torch
targets = torch.load('/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset/targets.pt', map_location='cpu', weights_only=False)
bin_labels = targets['binary_label']
import numpy as np
bin_labels = np.array(bin_labels)
print("Total:", len(bin_labels))
valid = ~np.isnan(bin_labels)
print("Valid:", valid.sum())
positives = np.sum(bin_labels[valid])
print("Positives:", positives)
print("Positive ratio:", positives / valid.sum())
