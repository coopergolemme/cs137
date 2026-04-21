import os
import torch

data_dir = '/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset'
targets_path = os.path.join(data_dir, 'targets.pt')
meta_path = os.path.join(data_dir, 'metadata.pt')

print("Loading targets...")
targets = torch.load(targets_path, map_location='cpu', weights_only=False)
print(f"Targets Type: {type(targets)}")

if isinstance(targets, torch.Tensor):
    print(f"Targets Shape: {targets.shape}")
elif isinstance(targets, dict):
    print(f"Targets dict len: {len(targets)}")
    example_key = list(targets.keys())[0]
    print(f"Example key: {example_key}, shape: {targets[example_key].shape}")

print("\nLoading metadata...")
meta = torch.load(meta_path, map_location='cpu', weights_only=False)
print(f"Meta Type: {type(meta)}")
if isinstance(meta, dict):
    print("Meta keys:", list(meta.keys()))

inps = os.path.join(data_dir, 'inputs', '2018')
inp_files = sorted(os.listdir(inps))
first_inp = os.path.join(inps, inp_files[0])
print(f"\nLoading {first_inp}...")
inp_tensor = torch.load(first_inp, map_location='cpu', weights_only=False)
print("Input shape:", inp_tensor.shape)
print("Input dtype:", inp_tensor.dtype)
