import yaml
def readConfig(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)