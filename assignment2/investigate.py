import torch
import torch.nn.functional as F
from dataset import WeatherDataset

ds = WeatherDataset(years_to_include=['2018'])

valid_count_drop = 0
valid_count_impute = 0
total_nans = 0
total_elements = 0

positives = 0
valid_bins = 0

for i in range(min(100, len(ds))):
    x, y = ds[i]
    
    nans_in_x = torch.isnan(x).sum().item()
    total_nans += nans_in_x
    total_elements += x.numel()
    
    has_nan = torch.isnan(x).any().item()
    if not has_nan:
        valid_count_drop += 1
        
    valid_count_impute += 1
    
    if not torch.isnan(y[6]).item():
        valid_bins += 1
        if y[6].item() > 0.5:
            positives += 1

print(f"Tested {min(100, len(ds))} samples")
print(f"NaN pixels: {total_nans} / {total_elements} ({total_nans/total_elements*100:.2f}%)")
print(f"Valid elements if dropping sample with any NaN: {valid_count_drop}")
print(f"Valid elements if imputing: {valid_count_impute}")
print(f"Positives in bin target: {positives} / {valid_bins} ({positives/max(1,valid_bins)*100:.2f}%)")
