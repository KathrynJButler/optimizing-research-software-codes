import os
import yaml


def load_config(config=None):
  if config is None:
    config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
  if not os.path.isfile(config):
    raise FileNotFoundError
  with open(config, 'r') as fp:
    return yaml.load(fp.read(), Loader=yaml.FullLoader)
