from integrator.utils import construct_data_loader, load_config

path = "/Users/luis/temp/data/test_yaml.yaml"
cfg = load_config(path)

data_loader = construct_data_loader(cfg)
