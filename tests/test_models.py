from integrator.configs.config_utils import (
    construct_yaml_configuration,
    generate_data_files,
    make_temp_data_dir,
)
from integrator.utils import dump_yaml_config

# write a temporary directory
data_dir = make_temp_data_dir()
# generate PyTorch data files for model
data_files = generate_data_files(data_dir=data_dir, save_files=True)
# generate a config.yaml for the model
cfg_dict = construct_yaml_configuration(data_dir=data_dir)
# save config.yaml
cfg_path = data_dir + "/test_yaml.yaml"

dump_yaml_config(
    cfg=cfg_dict,
    path=cfg_path,
)
