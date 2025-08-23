import os

import torch

path = "/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator/data/pass1/"

shoeboxes = torch.load(os.path.join(path,"standardized_shoeboxes.pt"))[:10000]
counts = torch.load(os.path.join(path, "raw_counts.pt"))[:10000]
metadata = torch.load(os.path.join(path, "metadata.pt"))[:10000]
dead_pixel_mask = torch.load(os.path.join(path, "masks.pt"))[:10000]
shoebox_features = torch.load(os.path.join(path, "shoebox_features.pt"))[:10000]

torch.save(shoeboxes,"standardized_shoeboxes_subset.pt")
torch.save(counts,"raw_counts_subset.pt")
torch.save(metadata,"metadata_subset.pt")
torch.save(dead_pixel_mask,"masks_subset.pt")
torch.save(shoebox_features,"shoebox_features_subset.pt")
