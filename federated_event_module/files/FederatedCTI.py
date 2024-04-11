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
import torch
from sklearn.model_selection import train_test_split
from datasample_generator import sample_generator
from pymisp import ExpandedPyMISP, PyMISP, MISPEvent, MISPAttribute
from datetime import datetime, timedelta
import pickle
import urllib3
import pause



FORMAT = "utf-8"
SIZE = 1024
# Environment Variables
HOST = os.environ.get('CENTRAL_HOST')
PORT = int(os.environ.get('CENTRAL_PORT'))
EPOCHS = int(os.environ.get('LOCAL_LIMIT_EPOCHS'))
BATCH_SIZE_ITER = int(os.environ.get('BATCH_SIZE'))
DATASET_TRAIN = os.environ.get('DATASET_TRAIN')
DATASET_INFERENCE = os.environ.get('DATASET_INFERENCE')
ORG_ID = os.environ.get('ORG_ID')
MISP_URL = os.environ.get('MISP_URL')
MISP_AUTH_KEY = os.environ.get('MISP_AUTH_KEY')
CLUSTERING_ALGO = os.environ.get('CLUSTERING_ALGO')


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class ClientHandler:
    def client_fl_program(self, host, port, dataset, epochs, batch_size, model, opt, loss):


        logging.info("Federated Client waiting for the organization ID.")
        # Block until the device id is found. Check every 30 seconds, indefinitely.
        CLIENT = ORG_ID
        logging.info(f"Federated Client got the following organization ID: {CLIENT}.")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"The training is running on: {device}.")



        cur_epoch = 0

        while True:
            try:
                client_socket = socket.socket()  # instantiate
                client_socket.connect((HOST, PORT))  # connect to the server

                # receive first message
                data = client_socket.recv(SIZE).decode(FORMAT)

                logging.info("Connected to server to send model.")

                # start training
                FILE_PATH = DATASET_TRAIN
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
                if cur_epoch % 50 == 0:
                    logging.info(" Saving the current model for backup...")
                    # Specify a path
                    PATH = os.getcwd()+"/file/Event_classifier_model.pt"
                    # Save
                    torch.save(model.state_dict(), PATH)

                if cur_epoch == epochs:

                    client_socket.close()
                    logging.info("Reaching the final training epoch. Closing the socket and saving the model...")
                    # Specify a path
                    PATH = os.getcwd()+"/file/Event_classifier_model.pt"
                    # Save
                    torch.save(model.state_dict(), PATH)
                    self.inference_flag = True
                    days = 1
                    logging.info("Total Training Epochs Done. Next update will be in " + str(days) + " days.")
                    week_from_now = datetime.now() + timedelta(days=days)
                    pause.until(datetime(week_from_now.year, week_from_now.month, week_from_now.day, week_from_now.hour))
                    logging.info(str(days) + " days have elapsed. Time for new model update.")
                    # Reset the current epoch
                    cur_epoch = -1

                cur_epoch += 1

            except Exception as e:
                logging.error(e)
                logging.error("Federated client program failed to start. Retrying in 30 seconds.")
                time.sleep(30)
                pass

    def client_inference_program(self, dataset, clustering_algo):


        logging.info("Lodding the test IoCs...")
        input_data_path = os.getcwd() + "/file/Dataset/input_"+ dataset +".xlsx"
        # Generate data samples from IoC files
        logging.info("Starting generating data samples from the IoCs")
        generated_samples, original_samples = sample_generator(input_data_path)


        data= generated_samples.values
        data = data[:, 0:66]
        data = Utils.normalize(data)
        num_class = 10
        num_feature = 66

        event_classes = ['APT', 'Attack', 'Backdoor', 'Botnet', 'Command and Control',
            'Exploitation', 'Malspam/Phishing', 'Malware', 'Ransomware', 'Trojan']

        logging.info("Test data samples are ready for the inference. Waiting for federated model...")

        while True:
            time.sleep(30)
            logging.info("Wake up to check the status...")
            if self.inference_flag == True:
                break

        while self.inference_flag == True:

            logging.info("Start the inference phase with current model...")

            logging.info("Loading the current federated event classification model...")
            # Load the pre-trained event classification model
            event_model = Models.Multi_Classifier_Reg(66, 168, 85, 45, 10)

            event_model.load_state_dict(torch.load(os.getcwd()+"/file/Event_classifier_model.pt"))

            # disable randomness, dropout, etc...
            event_model.eval()

            logging.info("Infering the classification model with data samples...")
            tensor_data = torch.FloatTensor(data.values)
            # predict with the model
            y_hat = event_model(tensor_data)
            _, event_pred = torch.max(y_hat, 1)
            event_pred = event_pred.tolist()

            logging.info("Loading the pre-trained threat level ranking model...")
            # Load the pre-trained L2R model
            l2r_model = torch.jit.load(os.getcwd()+'/file/Models/'+ dataset +'/l2rmodel.pt')
            l2r_model.eval()

            logging.info("Infering the threat level ranking model with data samples...")
            y_hat = l2r_model(tensor_data)
            level_pred = y_hat.detach().numpy()
            pred_max = np.max(level_pred)
            pred_min = np.min(level_pred)
            pred_range = pred_max - pred_min
            level_pred[(level_pred > (pred_max-(pred_range/4))) & (level_pred <= pred_max)] = 4
            level_pred[(level_pred > (pred_max-2*(pred_range/4))) & (level_pred < (pred_max-(pred_range/4))) ] = 3
            level_pred[(level_pred > (pred_max-3*(pred_range/4))) & (level_pred < (pred_max-2*(pred_range/4))) ] = 2
            level_pred[(level_pred >= pred_min) & (level_pred < (pred_max-3*(pred_range/4))) ] = 1
            level_pred = level_pred.squeeze()
            level_pred = level_pred.tolist()

            logging.info("Loading the pre-trained clustering model...")

            # Load the pre-trained clustering model
            with open(os.getcwd()+'/file/Models/'+ dataset +'/'+ clustering_algo +'.pkl', 'rb') as f:
                cluster_model = pickle.load(f)

            logging.info("Infering the clustering model with data samples...")
            cluster_predicts = cluster_model.fit_predict(data)


            try:
                logging.info("Connecting to MISP...")
                urllib3.disable_warnings()
                misp = ExpandedPyMISP(MISP_URL, MISP_AUTH_KEY, False)
                bookmark = 0
                while bookmark <= generated_samples.shape[0]:
                    for i in range(5):
                        instance = bookmark + i
                        r = misp.search(eventinfo=generated_samples.iloc[instance].loc[68], metadata=True)
                        cluster_id_str = "Cluster_ID:"+str(cluster_predicts[instance])
                        event_class = event_pred[instance]
                        event_class_str = "Event_class:"+str(event_classes[event_class])
                        threat_level = int(level_pred[instance])
                        if len(r) == 0:
                            event_obj = MISPEvent()
                            event_obj.distribution = 1
                            event_obj.threat_level_id = threat_level
                            event_obj.analysis = 1
                            event_obj.info = generated_samples.iloc[instance].loc[68]
                            event = misp.add_event(event_obj)
                            event_id, event_uuid = event['Event']['id'], event['Event']['uuid']
                            logging.info("Adding a new event with id:" + str(event_id) + " and UUID:"+str(event_uuid))
                            misp_attribute = MISPAttribute()
                            misp_attribute.value = str(original_samples.iloc[instance]['Atr_Value'])
                            misp_attribute.category = str(original_samples.iloc[instance]['Category'])
                            misp_attribute.type = str(original_samples.iloc[instance]['Atr_type'])
                            misp_attribute.comment = str(original_samples.iloc[instance]['Comment'])
                            misp_attribute.to_ids = str(original_samples.iloc[instance]['Is_IDS'])
                            r = misp.add_attribute(event_uuid, misp_attribute)
                            tag = misp.tag(event_uuid, cluster_id_str)
                            tag = misp.tag(event_uuid, event_class_str)
                            logging.info("Adding a cluster tag:" + cluster_id_str)
                            logging.info("Adding an event tag:" + event_class_str)
                            bookmark = instance


                        else:
                            for obj in r:
                                misp_attribute = MISPAttribute()
                                misp_attribute.value = str(original_samples.iloc[instance]['Atr_Value'])
                                misp_attribute.category = str(original_samples.iloc[instance]['Category'])
                                misp_attribute.type = str(original_samples.iloc[instance]['Atr_type'])
                                misp_attribute.comment = str(original_samples.iloc[instance]['Comment'])
                                misp_attribute.to_ids = str(original_samples.iloc[instance]['Is_IDS'])
                                r = misp.add_attribute(str(obj['Event']['uuid']), misp_attribute)
                                tags = []
                                for tag in obj['Event']['Tag']:
                                    tags.append(tag['name'])

                                if cluster_id_str not in tags:
                                    tag = misp.tag(event_uuid, cluster_id_str)
                                    logging.info("Adding a new tag:" + cluster_id_str + " to the event:"+str(event_uuid))

                                logging.info("Add a new attribute to the event:" + str(event_id))
                            bookmark = instance

                        time.sleep(5)

            except Exception as e:
                logging.error(e)
                logging.error("Connecting to MISP failed, Reconnecting in 30 minutes.")
                time.sleep(30)
                pass




    def __init__(self, host, port, dataset_train, dataset_inference, epochs, batch_size, model, opt, loss, clustering_algo):
        """
        TODO: Description
        """
        self.global_weights = []
        self.external_xsyn = []
        self.external_ysyn = []
        self.inference_flag = False
        # Initiate federated client thread
        self.client_fl_program = threading.Thread(target=self.client_fl_program, args=([host, port, dataset_train, epochs, batch_size, model, opt, loss]))
        self.client_inference_program = threading.Thread(target=self.client_inference_program, args=([dataset_inference, clustering_algo]))
        self.client_fl_program.start()
        self.client_inference_program.start()

if __name__=='__main__':

    coloredlogs.install(fmt='%(asctime)s,%(msecs)03d %(name)s[%(process)d] %(levelname)s %(message)s')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    """ Define initial model """
    model = Models.Multi_Classifier_Reg(66, 168, 85, 45, 10)
    opt = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)
    loss = nn.MSELoss()

    client=ClientHandler(HOST, PORT, DATASET_TRAIN, DATASET_INFERENCE, EPOCHS, BATCH_SIZE_ITER , model, opt, loss, CLUSTERING_ALGO)
