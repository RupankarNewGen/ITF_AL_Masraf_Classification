from configparser import ConfigParser
import os

# config snippet
configur = ConfigParser()
conf_folder_path: str = "src/main/extraction/config"
config_file_name: str = "prod.ini"

if os.path.exists(f"{conf_folder_path}/{config_file_name}"):
    configur.read(f"{conf_folder_path}/{config_file_name}")

gv_key = configur['LetterOfCredit']['PerformaInvoice']
print(gv_key)