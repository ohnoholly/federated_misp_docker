# Imports
import os
import json
import socket
import threading
import coloredlogs, logging
import time
import argparse
import numpy as np
from json import JSONEncoder
import MLUnit
import Utils
import Models
from torch import optim, nn
from config_reader import Configs
import torch
from sklearn.model_selection import train_test_split

coloredlogs.install(fmt='%(asctime)s,%(msecs)03d %(name)s[%(process)d] %(levelname)s %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

SIZE = 1024
FORMAT = "utf-8"
# Environment Variables
HOST = os.environ.get('CENTRAL_HOST')
PORT = int(os.environ.get('CENTRAL_PORT'))
EPOCHS = int(os.environ.get('LOCAL_LIMIT_EPOCHS'))
BATCH_SIZE_ITER = int(os.environ.get('BATCH_SIZE'))
DATASET = os.environ.get('DATASET')


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class ClientHandler:
    def client_program(self, host, port, dataset, epochs, batch_size, model, opt, loss):

        # Create a configurations reader instance

        self.configs_dir = os.environ.get("CONFIG_DIR", "")
        self.conf_reader = Configs(self.configs_dir)

        logging.info("Federated Client waiting for the device ID.")
        # Block until the device id is found. Check every 30 seconds, indefinitely.
        CLIENT = self.conf_reader.waitForDeviceId(n_tries=1, time_interval=3)
        logging.info(f"Federated Client got the following device ID: {CLIENT}.")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)


        if dataset =='CIRCL':
            DATA_FILE = "CIRCL_DS_0.npz"
        elif dataset == 'BOTVRIJ':
            DATA_FILE = "BOTVRIJ_DS_0.npz"
        elif dataset == 'MIX':
            DATA_FILE = "MISP_DS_0.npz"

        cur_epoch = 0

        while True:
            try:
                client_socket = socket.socket()  # instantiate
                client_socket.connect((HOST, PORT))  # connect to the server

                # receive first message
                data = client_socket.recv(SIZE).decode(FORMAT)
                epochs = int(data)

                logging.info("Connected to server to send model.")

                # start training
                FILE_PATH = DATA_FILE
                npzfile = np.load(FILE_PATH, allow_pickle = True)
                x_train = npzfile["train_x"]
                y_train = npzfile["train_y"]
                x_test = npzfile["test_x"]
                y_test = npzfile["test_y"]
                num_class = npzfile["num_class"]
                x_train = torch.tensor(x_train)
                y_train = torch.tensor(y_train)
                x_test = torch.tensor(x_test)
                y_test = torch.tensor(y_test)
                num_class = int(num_class)

                logging.info(f"x_train shape: {x_train.shape}")
                logging.info(f"y_train shape: {y_train.shape} ")


                model, weights, xsyn, ysyn = MLUnit.train(x_train, y_train, x_test, y_test,
                                                    model, device, opt, loss, num_class,
                                                    self.external_xsyn, self.external_ysyn,
                                                    BATCH_SIZE_ITER, self.global_weights)

                data = f"Local Model_{CLIENT}"
                client_socket.send(data.encode(FORMAT))

                out_xsyn, _, out_ysyn, _ = train_test_split(xsyn, ysyn, test_size=0.6)

                message = json.dumps({"weights": weights, "xsyn": out_xsyn, "ysyn": out_ysyn}, cls=NumpyArrayEncoder)
                FILENAME = f"{CLIENT}-weights.json"
                with open(FILENAME, "w") as outfile:
                    outfile.write(message)

                FILESIZE = os.path.getsize(FILENAME)

                data = f"{FILENAME}_{FILESIZE}_{CLIENT}"
                client_socket.send(data.encode("utf-8"))

                msg = client_socket.recv(SIZE).decode("utf-8")
                logging.info(f"{msg}")

                """ Data transfer. """
                with open(FILENAME, "r") as f:
                    while True:
                        data = f.read(SIZE)
                        if not data:
                            break
                        client_socket.send(data.encode(FORMAT))
                        msg = client_socket.recv(SIZE).decode(FORMAT)
                logging.info(msg)

                os.remove(FILENAME)
                client_socket.close()

                client_socket = socket.socket()  # instantiate
                client_socket.connect((HOST, PORT))  # connect to the server
                logging.info("Connected to server to receive model.")

                """ Global model transfer """
                data = client_socket.recv(SIZE).decode(FORMAT)
                epochs = int(data)

                data = f"Model Update_{CLIENT}"
                client_socket.send(data.encode(FORMAT))

                data = client_socket.recv(SIZE).decode(FORMAT)
                item = data.split("_")
                FILENAME = item[0]
                FILESIZE = int(item[1])

                logging.info("Filename and filesize received from server.")
                client_socket.send("Filename and filesize received.".encode(FORMAT))

                """ Data transfer """
                logging.info("Data Transfer Started from server.")
                with open(f"recv_{FILENAME}", "w") as f:
                    while True:
                        data = client_socket.recv(SIZE).decode(FORMAT)
                        if not data:
                            break
                        f.write(data)
                        client_socket.send("Global Model Received.".encode(FORMAT))

                data = open(f"recv_{FILENAME}")
                data = json.load(data)
                os.remove(f"recv_{FILENAME}")
                self.global_weights = data['weights']
                self.global_weights = np.array(self.global_weights, dtype=object)
                self.external_xsyn = data['xsyn']
                self.external_ysyn = data['ysyn']


                logging.info("Testing Model")
                MLUnit.test(model, self.global_weights, x_test, y_test)

                logging.info("Current Epoch " + str(cur_epoch)+ " ends.")
                if cur_epoch == epochs-2:
                    client_socket.close()
                    logging.info("Reaching the final training epoch. Saving the model...")
                    # Specify a path
                    PATH = "Event_classifier_model.pt"
                    # Save
                    torch.save(model.state_dict(), PATH)

                cur_epoch += 1

            except Exception as e:
                logging.error(e)
                logging.error("Federated client program failed to start. Retrying in 30 seconds.")
                time.sleep(30)
                pass

    def __init__(self, host, port, dataset, epochs, batch_size, model, opt, loss):
        """
        TODO: Description
        """
        self.global_weights = []
        self.external_xsyn = []
        self.external_ysyn = []
        # Initiate federated client thread
        self.client_program = threading.Thread(target=self.client_program, args=([host, port, dataset, epochs, batch_size, model, opt, loss]))
        self.client_program.start()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    args = parser.parse_args()

    """ Define initial model """
    model = Models.Multi_Classifier_Reg(66, 168, 85, 45, 10)
    opt = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)
    loss = nn.MSELoss()

    client=ClientHandler(HOST, PORT, DATASET, EPOCHS, BATCH_SIZE_ITER , model, opt, loss)
