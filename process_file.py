from utils import batch_process_pdf, load_config


def process_file(input_file: str):
    config = load_config(config_path="config.yaml")
    batch_process_pdf(input_file, config)

