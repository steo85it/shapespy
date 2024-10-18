import logging
import os
import yaml

logging.basicConfig(level=logging.WARN)

class SfsOpt:

    _instance = None  # Class attribute to hold the singleton instance

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(SfsOpt, cls).__new__(cls)
            # Ensure that kwargs are handled in __init__ or here directly if __init__ isn't flexible.
            cls._instance.__init__(**kwargs)
        return cls._instance

    def __init__(self, **kwargs):
        if not hasattr(self, 'initialized'):  # Guard to prevent re-initialization
            self.__dict__.update(kwargs)
            self.load_wkt_configs()
            self.initialized = True

    def load_wkt_configs(self):
        """Load configurations that end with '.wkt'."""
        for key, value in list(self.__dict__.items()):
            if isinstance(value, str) and value.endswith('.wkt'):
                with open(value, 'r', encoding='utf-8') as file:
                    self.__dict__[key] = file.read()

    @staticmethod
    def get_instance():
        # if not SfsOpt._instance:
        #     raise ValueError("SfsOpt instance is not yet created")
        if SfsOpt._instance is None:
            SfsOpt()
        return SfsOpt._instance

    def update_config(self, **config):
        self.__dict__.update(config)
        # for key, value in config.items():
        #     setattr(self, key, value)  # Dynamically set attributes based on the key-value pairs
        self.load_wkt_configs()

    def display(self):
        for key in self.__dict__:
            print(f"{key} = {getattr(self, key)}")

    @staticmethod
    def check_consistency():
        return

    def get(self, name):
        return self.__dict__[name]

    def set(self, name, value):
        self.__dict__[name] = value
        print(f"### SfsOpt.{name} updated to {value}.")

    def to_yaml(self, file_path):
        """Dumps the configuration dictionary to a YAML file."""
        with open(file_path, 'w') as file:
            yaml.dump(self.__dict__, file, default_flow_style=False)
        print(f"Configuration saved to {file_path}")

    def from_yaml(file_path):
        """Reads the configuration dictionary from a YAML file."""
        with open(file_path, 'r') as file:
            loaded_config = yaml.safe_load(file)
        return SfsOpt(**loaded_config)            
       
    @staticmethod
    def to_dict():
        return SfsOpt.__dict__

    @staticmethod
    def clone(opts):
        SfsOpt.__conf = opts.copy()


if __name__ == '__main__':

    print(SfsOpt.get("rootdir"))

    opt = SfsOpt()
    print(opt.get("isisdir"))

    print(opt.get("local"))
    opt.set("local", False)

    opt.check_consistency()
    print(opt.get("isisdir"))

    opt.display()
