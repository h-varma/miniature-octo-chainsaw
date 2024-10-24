import logging


def create_logger(name: str, level=logging.INFO, output_file=True, output_console=True):
    """
    Create a logger object.

    Parameters
    ----------
    name : str
        name of the logger
    level : int
        logging level
    output_file : bool
        whether to output to a file or not
    output_console : bool
        whether to output to console or not

    Returns
    -------
    logging.Logger : logger object
    """
    logger_ = logging.getLogger(name)
    logger_.setLevel(level=level)

    format_style = "[%(asctime)s] {%(module)s:%(lineno)d} %(levelname)s - %(message)s"
    date_format = "%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=format_style, datefmt=date_format)

    # log into a file
    file_handler = logging.FileHandler("{}.log".format(name))
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
