import yaml
from jinja2 import Template

def load_config_yaml(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # safe_load prevents execution of arbitrary code
    return config

def load_config_yaml_with_tileid(file_path, args):
    # Read the template YAML file
    with open(file_path, 'r') as file:
        template = Template(file.read())

    # Render the template with the value of X from command-line arguments
    rendered_yaml = template.render(tileid=args.tileid)

    # Parse the rendered YAML to ensure it is correct
    config = yaml.safe_load(rendered_yaml)

    return config
