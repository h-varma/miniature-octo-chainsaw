import logging


def create_logger(name: str, level=logging.INFO, output_file=True, output_console=True):
    logger_ = logging.getLogger(name)
    logger_.setLevel(level=level)

    formatter = logging.Formatter('[%(asctime)s] {%(module)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')

    # log into a file
    file_handler = logging.FileHandler('{}.log'.format(name))
    file_handler.setFormatter(formatter)

    # log into console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    if output_file:
        logger_.addHandler(file_handler)
    if output_console:
        logger_.addHandler(stream_handler)

    return logger_


logger = create_logger(__name__, output_file=False)
