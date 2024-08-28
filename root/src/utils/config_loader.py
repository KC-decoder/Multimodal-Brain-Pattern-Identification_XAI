import yaml
import os

def get_config_path():
    # Get the absolute path to the src directory
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Traverse upwards until you reach the root directory that contains 'config'
    while not os.path.exists(os.path.join(src_dir, 'config/config.yml')):
        src_dir = os.path.dirname(src_dir)
        
    # Construct the absolute path to the config file
    return os.path.join(src_dir, 'config/config.yml')


def load_config():
    config_path = get_config_path()
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Get the actual root_dir from the config
    root_dir = cfg.get('root_dir')

    # Replace ${root_dir} in all paths with the actual root directory path
    for key, value in cfg.items():
        if isinstance(value, str) and "${root_dir}" in value:
            cfg[key] = value.replace("${root_dir}", root_dir)

    return cfg