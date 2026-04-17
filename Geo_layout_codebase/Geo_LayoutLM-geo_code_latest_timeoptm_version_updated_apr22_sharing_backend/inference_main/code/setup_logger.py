import logging.handlers
import os
import logging
from datetime import datetime
import atexit
import sys

# Enter False to only log warning and above log statements
debug_mode = True
non_debug_mode_log_level = logging.INFO



class CustomLogger:
    def __init__(self, log_file_max_size: int = 20, log_folder_name: str = "OTHER_UTILITY_LOGS", take_complete_log_folder_path: bool = False) -> None:
        self.log_file_max_size = log_file_max_size
        self.logger_object = None
        log_folder_name = log_folder_name.strip("/")
        if not take_complete_log_folder_path:
            self.log_dir = os.path.join(os.getcwd(), f"../logs/Api_Call_Logs/{log_folder_name}/")
        else:
            self.log_dir = os.path.abspath(f"{log_folder_name}/")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        
        if not hasattr(self, "_atexit_registered"):
            atexit.register(self.cleanup)
            self._atexit_registered = True

    def cleanup(self):
        if self.logger_object is not None:
            try:
                handlers = self.logger_object.handlers[:]
                for handler in handlers:
                    self.logger_object.removeHandler(handler)
                    handler.close()
                self.logger_object.debug(f"LOG FILE HANDLER REMOVED FOR: {self.unique_log_file}")
            except Exception as e:
                print(f"Cleanup exception: {e}")

    def generate_logger_object(self, log_file_name: str = "", ignore_time: bool = False):
        currrent_time = f'_{str(datetime.strftime(datetime.now(), "%d_%m_%Y_%H_%M_%S"))}'
        if log_file_name == "": log_file_name: str = "general_logs"; ignore_time = True
        if ignore_time: currrent_time = ""

        self.unique_log_file = str(os.path.join(self.log_dir, f"{str(log_file_name)}{currrent_time}.log"))
        self.unique_log_file = self.unique_log_file.replace("'", "").replace('"', "")

        logs = logging.getLogger(f"{str(log_file_name)}{currrent_time}")
        # Clear existing handlers
        for handler in logs.handlers[:]:
            logs.removeHandler(handler)
            handler.close()

        logs.setLevel(logging.DEBUG if debug_mode else non_debug_mode_log_level)
        formatter = logging.Formatter('%(asctime)s %(levelname)s Line No:%(lineno)d %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        handler = logging.handlers.RotatingFileHandler(
            filename=self.unique_log_file,
            mode='a',
            maxBytes=int(self.log_file_max_size) * 1024 * 1024,
            backupCount=2,
            delay=False
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logs.addHandler(handler)

        logs.info("-----------------------------------------------------------------------------------------")
        logs.info("                             NEWGEN SOFTWARE TECHNOLOGIES LIMITED")
        logs.info("* Group: Number Theory")
        logs.info("* Product/Project: Intelligent Trade Finance")
        logs.info("-----------------------------------------------------------------------------------------\n")
        logs.info(f"############################ LOGGING FILE : {self.unique_log_file} #############################\n")

        self.logger_object = logs
        return self

    def __del__(self):
        if sys.meta_path is not None:
            self.cleanup()
