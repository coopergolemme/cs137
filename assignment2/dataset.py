import os
import torch
from torch.utils.data import Dataset
import numpy as np
import datetime

class WeatherDataset(Dataset):
    def __init__(self, data_dir='/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset', years_to_include=['2018', '2019', '2020']):
        self.data_dir = data_dir
        self.inputs_dir = os.path.join(data_dir, 'inputs')
        
        # Load targets
        targets_path = os.path.join(data_dir, 'targets.pt')
        self.targets = torch.load(targets_path, map_location='cpu', weights_only=False)
        
        # Build a mapping from time string '%Y%m%d%H' to the index in targets arrays
        self.time_to_idx = {}
        target_times = np.array(self.targets['time'])
        
        # Convert np.datetime64 to datetime.datetime directly
        # Sometimes it's int array (nanoseconds). If so, cast to datetime64[ns]
        if target_times.dtype != 'datetime64[ns]':
            target_times = target_times.astype('datetime64[ns]')
        dt_times = target_times.astype('M8[us]').astype(datetime.datetime)
        
        for idx, dt in enumerate(dt_times):
            time_str = dt.strftime('%Y%m%d%H')
            self.time_to_idx[time_str] = idx

        self.samples = []
        for year in years_to_include:
            year_dir = os.path.join(self.inputs_dir, str(year))
            if not os.path.exists(year_dir):
                print(f"Directory not found: {year_dir}")
                continue
            
            files = sorted([f for f in os.listdir(year_dir) if f.startswith('X_') and f.endswith('.pt')])
            for file in files:
                # file: X_YYYYMMDDHH.pt
                time_str = file.replace('X_', '').replace('.pt', '')
                try:
                    dt = datetime.datetime.strptime(time_str, '%Y%m%d%H')
                    dt_target = dt + datetime.timedelta(hours=24)
                    target_time_str = dt_target.strftime('%Y%m%d%H')
                except ValueError:
                    continue
                
                # Check if we have targets for t+24
                if target_time_str in self.time_to_idx:
                    inp_path = os.path.join(year_dir, file)
                    target_idx = self.time_to_idx[target_time_str]
                    self.samples.append((inp_path, target_idx))
                    
        all_vals_cont = []
        for inp_path, target_idx in self.samples:
            all_vals_cont.append(self.targets['values'][target_idx])
            
        if len(all_vals_cont) > 0:
            all_vals_cont_arr = np.array(all_vals_cont)
            self.target_mean = torch.tensor(np.nanmean(all_vals_cont_arr, axis=0), dtype=torch.float32)
            self.target_std = torch.tensor(np.nanstd(all_vals_cont_arr, axis=0) + 1e-6, dtype=torch.float32)
        else:
            self.target_mean = torch.zeros(6, dtype=torch.float32)
            self.target_std = torch.ones(6, dtype=torch.float32)
            
        print(f"Initialized WeatherDataset with {len(self.samples)} samples for years {years_to_include}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        inp_path, target_idx = self.samples[idx]
        
        # Load Input Data
        X = torch.load(inp_path, map_location='cpu', weights_only=False)
        X = X.to(torch.float32)
        
        # Note: We keep NaNs here so we can filter these samples out dynamically in the training loop.
        # X = torch.where(torch.isnan(X), torch.zeros_like(X), X)

        # Permute (H, W, C) -> (C, H, W)
        if len(X.shape) == 3 and X.shape[2] == 42:
            X = X.permute(2, 0, 1)
            
        # Get targets
        vals_cont = self.targets['values'][target_idx]  # Shape [6]
        val_bin = self.targets['binary_label'][target_idx]  # Shape [1] or scalar
        
        # Convert to tensor if they are numpy
        if not isinstance(vals_cont, torch.Tensor):
            vals_cont = torch.tensor(vals_cont, dtype=torch.float32)
        else:
            vals_cont = vals_cont.to(dtype=torch.float32)
            
        # Normalize continuous values
        vals_cont = (vals_cont - self.target_mean) / self.target_std
            
        if not isinstance(val_bin, torch.Tensor):
            val_bin = torch.tensor([val_bin], dtype=torch.float32)
        else:
            val_bin = val_bin.view(1).to(dtype=torch.float32)
            
        # Concatenate: final target is shape [7]
        Y = torch.cat([vals_cont, val_bin], dim=0)
        
        return X, Y

if __name__ == '__main__':
    ds = WeatherDataset(years_to_include=['2018'])
    print("Dataset length:", len(ds))
    if len(ds) > 0:
        X, Y = ds[0]
        print("X shape:", X.shape)
        print("Y shape:", Y.shape)
        print("Y preview:", Y)
