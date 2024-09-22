import yaml
import os

# Base configuration 
base_config = {
    "epochs": 800,
    "dmodel": 64,
    "batch_size": 200,
    "val_split": 0.2,
    "test_split": 0.1,
    "num_workers": 1,
    "include_test": False,
    "subset_size": 1000,
    "single_sample_index": None,
    "C": 6,
    "Z": 3,
    "H": 21,
    "W": 21,
    "rank": 5,
    "depth": 10,
    "feature_dim": 7,
    "dropout": None,
    "input_dim": 64,
    "channels": 3,
    "height": 21,
    "width": 21,
    "p_I_scale": 0.0001,
    "p_bg_scale": 0.0001,
    "p_I": {
        "distribution": "Exponential",
        "rate": 0.1
    },
    "p_bg": {
        "distribution": "Exponential",
        "rate": 0.1
    },
    "q_I": {
        "distribution": "Gamma"
    },
    "q_bg": {
        "distribution": "Gamma"
    },
    "accelerator": "gpu",
    "precision": 32,
    "learning_rate": 0.001,
    "total_steps": None,
    "shoebox_data": "simulated_hewl_816/simulated_samples.pt",
    "metadata": "simulated_hewl_816/simulated_metadata.pt",
    "dead_pixel_mask": "simulated_hewl_816/simulated_masks.pt"
}

# Configurations
configs = [
    {"name": "CNNResNet_MVNProfile", "encoder_type": "CNNResNet", "profile_type": "MVNProfile"},
    {"name": "CNNResNet_SoftmaxProfile", "encoder_type": "CNNResNet", "profile_type": "SoftmaxProfile"},
    {"name": "FcResNet_MVNProfile", "encoder_type": "FcResNet", "profile_type": "MVNProfile"},
    {"name": "FcResNet_SoftmaxProfile", "encoder_type": "FcResNet", "profile_type": "SoftmaxProfile"}
]

# Ensure the config directory exists
os.makedirs("config", exist_ok=True)

# to handle NONE in yaml file
def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', 'null')

yaml.SafeDumper.add_representer(type(None), represent_none)

# Generate configs
for config in configs:
    full_config = base_config.copy()
    full_config.update(config)
    
    filename = f"config/{config['name']}_config.yaml"
    with open(filename, 'w') as f:
        yaml.safe_dump(full_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated: {filename}")

print("All YAML configs generated successfully!")
