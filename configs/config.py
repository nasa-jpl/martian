import yaml
import os


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self._load_config()

    def _load_config(self):
        """Loads the configuration from the file."""
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def update_config(self, new_values):
        """Updates the configuration values in memory."""
        for key, value in new_values.items():
            if key in self.config_data:
                self.config_data[key] = value
            else:
                raise KeyError(f"{key} not found in the configuration")
            
    def get_config(self):
        """Returns the current configuration as a dictionary."""
        return self.config_data

    def set_config(self, config):
        """Allows setting the entire configuration from a dictionary."""
        self.config_data = config

    def save_config(self, new_config_file=None):
        """Saves the configuration to a specified file.
        If no file is specified, it saves to the original file."""
        if new_config_file is None:
            new_config_file = self.config_file
        with open(new_config_file, 'w') as file:
            yaml.safe_dump(self.config_data, file)
