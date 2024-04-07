
from math import log
from pathlib import Path
import json
from time import sleep
import logging
import os
from pprint import pprint



class Configs(object):

    def __init__(self, configDirectory = "") -> None:
        self.configDirectory = configDirectory

        self.configFilePath = ""
        if len(self.configDirectory) != 0:
            self.configFilePath = f"{self.configDirectory}/"

        self.configFilePath += "config.json"
        self.path = Path(self.configDirectory)
        self.configPath = Path(self.configFilePath)
        logging.info(f"ConfigReader created for path {self.configPath.absolute()}")
        self.configs = None
        self.env_client_id = None # client id from an environment variable

    def directoryExists(self) -> bool:
        return self.path.is_dir()

    def configFileExists(self) -> bool:
        return self.configPath.is_file()

    def readConfigJson(self) -> dict:
        #if not self.configs:

        f = self.configPath.read_text()
        try:
            self.configs = json.loads(f)
        except json.JSONDecodeError:
            logging.warning('Configurations file is empty or not in a JSON format!')
            return None
        return self.configs

    def getDeviceIdFromEnvVariable(self) -> str:
        if self.env_client_id is not None:
            return self.env_client_id

        py_serv_arg = os.environ.get('PYTHON_SERVICE_ARGUMENT', None)
        if py_serv_arg is None:
            self.env_client_id = os.environ.get("CLIENT_ID", None)
        else:
            py_serv_arg = json.loads(py_serv_arg)
            try:
                self.env_client_id = py_serv_arg["CLIENT_ID"]
            except:
                self.env_client_id = None
        return self.env_client_id


    def getDeviceIdFromFile(self) -> str:
        if(not self.directoryExists()):
            return None

        logging.info("Found configurations directory")
        if(not self.configFileExists()):
            return None
        logging.info("Found configurations file")
        confs = self.readConfigJson()
        logging.info("Read configurations file, with contents: ")
        pprint(confs)
        if confs:
            return confs.get("device_id", None)

        return None

    def getDeviceId(self, from_file=True, from_environ = True):
        device_id = None
        if(from_file):
            # try to read from file
            device_id = self.getDeviceIdFromFile()
            logging.info(f"Found the device ID '{device_id}' in the file")

        if(from_environ and device_id is None):
            # try to read from the environment variable if it was not possible to read from the file
            device_id = self.getDeviceIdFromEnvVariable()
            if(device_id is not None):
                logging.info(f"Found the device ID '{device_id}' in the environment variables", )

        return device_id

    def readDeviceId(self, time_interval, from_file=True, from_environ=True):
        sleep(time_interval)
        return self.getDeviceId(from_file=from_file, from_environ=from_environ)

    def waitForDeviceId(self, n_tries = 1, time_interval = 1, from_file=True, from_environ=True) -> str:
        dev_id = None
        curr_tries = 0
        if n_tries == -1:
            while(dev_id is None):
                dev_id = self.readDeviceId(time_interval, from_file=from_file, from_environ=from_environ)
                logging.info(f"Searching for Device ID, indefinitely. Another try...")
        else:
            while(curr_tries < n_tries and dev_id is None):
                dev_id = self.readDeviceId(time_interval, from_file=from_file, from_environ=from_environ)
                curr_tries += 1
                logging.info(f"Searching for Device ID, indefinitely. Try {curr_tries} of {n_tries}")

        return dev_id
