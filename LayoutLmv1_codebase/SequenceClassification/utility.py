import logging
import os

def set_basic_config_for_logging(folder_path: str = None, filename: str = None):
    """
    Set the basic config for logging python program.
    :return: None
    """

    if folder_path is None:
        folder_path = ""

    log_file_path = os.path.join(folder_path, f"{filename}.log")
    print(f"log file path is {log_file_path}")
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(message)s',
                        filemode='w')
    


def get_logger_object_and_setting_the_loglevel(log_level: str = "INFO"):
    """
    get the logger object and set the loglevel for the logger object
    :return: Logger Object
    """
    # Creating an object
    logger_object = logging.getLogger()
    level = None

    if log_level == "INFO":
        level = logging.INFO
    elif log_level == "DEBUG":
        level = logging.DEBUG
    elif log_level == "CRITICAL":
        level = logging.CRITICAL
    else:
        level = logging.ERROR

    # Setting the threshold of logger to DEBUG
    logger_object.setLevel(level)
    return logger_object