# Read from config.json file

import json

def load_config_multilayer():
    with open('config.json') as json_file:
        config = json.load(json_file)
        return config['multilayer']