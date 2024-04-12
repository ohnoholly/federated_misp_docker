# Federated CTI

The repository is forked from [misp-docker](https://github.com/MISP/misp-docker). It contains three main containers, one is MISP-core, one is MISP-module, and the last one is the main container for the federated CTI module. Please follow the steps to install the  containers.

The federated CTI module consists of two phrases running as two threads. First is the training phase. It is implemented with [HSphereSMOTE](https://github.com/ohnoholly/P2P-FedAggregation) to rebalance the training dataset and generate synthetic data to share with other clients. Second phase is the Inference phase. Once the federated event classifcation model is finished training, it will trigger this thread to infer three ML modules (i) Federated event classification model, (ii) Pre-trainned threat level ranking model, and (iii) Pre-trainned IoC clustering model. The final generated IoCs will be uploaded to the local MISP instance (GUI), which is the misp-core in this repository.

## Installation
Please first check the `enviorment.sh` file in `/federated_event_module/files/` dictionary. 
* Plese change the variable of `ORG_ÌD` to your organization (IPN or 1Global). 
* The variable of `MISP_AUTH_KEY` can be changed to any 40 characters long SHA-1 hash. The default hash is the SHA-hash from 'arcadianiot'. Please note that if this variable is changed, please also change the value of `ADMIN_KEY` in `template.env` file in the root dictionary of the repository.  
* The variable of `MISP_URL` possiblely is needed to be changed after the MISP-core is up and running and the docker network is settled. This will be descirbled more in the following. 

Please go back to the root level of the repository where you can find docker-compose file.
Copy the `template.env` to `.env`,  build the containers via docker compose, and start installing the containers.

```sh
cd federated_misp_docker
cp template.env .env
docker compose build --no-cache
docker compose up
```
When the message from MISP-core is shown as 'The MISP is ready to log in', please stop the federated-cti container for a while from another window of terminal. 
Please check the docker network by using following command.

```sh
docker network ls
```
And please inspect the network created by this repository, should be called 'federated_misp_docker_inner_network'  by using:
```sh
docker network inspect federated_misp_docker_inner_network
```
Please find the IP address used by MISP-core, and change the address for the variable of `MISP_URL` in the `enviorment.sh` file in `/federated_event_module/files/` dictionary.
After it is done, please go back to the root level of the repository and rebuld the federated-cti container by using following command:
```sh
docker compose up -d --no-deps --build federated-cti
```
If there is no logs are showed. Please go back to the orignal window or use the following command to check the logs:
```sh
docker logs <container ID> --follow
```
The training and the inference should be running successfully!

