from sample_weights import Weighting, conf_to_log
import sys
import logging as lg
import os

if __name__ == '__main__':
    conf_path = sys.argv[1]
    log_path = conf_to_log(conf_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # Configure logging to write to a file
    lg.basicConfig(
        level=lg.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        filename=f'{log_path}/info.log',
        encoding='utf-8'
    )

    lg.info('Start')
    Weighting(config_file=conf_path, log_dir= log_path)