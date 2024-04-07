import numpy as np
import logging

import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
import HSphereSMOTE



def train_classifier(cl, opt, loss, x, y, device):
    x.to(device)
    y.to(device)
    #Reset gradients
    opt.zero_grad()
    #Train on real data
    pred = cl(x)
    pred.to(device)
    pred = torch.squeeze(pred, 1)
    err = loss(pred, y).to(device)
    err.backward()

    #Update optimizer
    opt.step()
    return err, pred

def predict(net, Xtest, device):
    net.to(device)
    Xtest.to(device)
    with torch.no_grad():
        Ypred = net.forward(Xtest).to(device)
        label = Ypred > 0.5
    return Ypred, label

def train_user(cl, opt, loss, x, y, epochs, batch_size, device):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    for i in range(epochs):
            #Shuffle training data
            logging.info("Training for epoch " + str(i))
            r = torch.randperm(x.size()[0])
            x = x[r]
            y = y[r]
            for beg_i in range(0, x.size(0), batch_size):
                xpo = Variable(x[beg_i:beg_i + batch_size, :]).to(device)
                ypo = Variable(y[beg_i:beg_i + batch_size]).to(device)
                err, pred = train_classifier(cl, opt, loss, xpo, ypo, device)
    return err, pred

def train(x_train, y_train, x_test, y_test, model, device, opt, loss,  num_class, e_xsyn, e_ysyn, train_batch_size, global_weights):

    if not e_xsyn:
        e_xsyn = np.empty([0, x_train.size(1)])
        e_ysyn = np.empty([0, y_train.size(1)])

    # Merge the training data with received synthetic data
    x_train = np.concatenate([x_train, e_xsyn])
    y_train = np.concatenate([y_train, e_ysyn])
    logging.info("Data Shape Before HSphere: X_Train: " + str(x_train.shape) + " Y_Train: " + str(y_train.shape))
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    xsyn, ysyn = HSphereSMOTE.Sampling_Multi(x_train, y_train, num_class, ratio=0.8, k=10)
    xsyn = xsyn.numpy()
    ysyn = ysyn.numpy()

    x_train_client = np.concatenate([x_train, xsyn])
    y_train_client = np.concatenate([y_train, ysyn])

    logging.info("Data Shape After HSphere: X_Train: " + str(x_train_client.shape) + " Y_Train: " + str(y_train_client.shape))


    input_dim = x_train.shape[1]

    x_train = np.asarray(x_train_client)
    y_train = np.asarray(y_train_client)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    model = model.to(device)

    params = model.named_parameters()
    dict_params = dict(params)

    if len(global_weights) != 0:
        i = 0
        for name1, param1 in params:
            dict_params[name1].data.copy_(torch.Tensor(global_weights[i]))
            i += 1

    partial_epochs = 5


    error, pred = train_user(model, opt, loss, x_train, y_train, partial_epochs,train_batch_size, device)

    acc, fscore_macro = computeTestErrors(model, x_test, y_test, device)

    logging.info("Training Done, AUC Score: " + str(acc))

    weights = []
    params = model.named_parameters()

    for name, param in params:
        weights.append(param.detach().numpy())

    return model, weights, xsyn, ysyn

def computeTestErrors(cl,Xtst,Ytst, device):

    logging.info("Computing test errors.")
    Xtst = Xtst.astype(np.float32)
    Ytst = Ytst.astype(np.float32)
    Xtst = torch.from_numpy(Xtst)
    Ytst = torch.from_numpy(Ytst)

    with torch.no_grad():
        err = 0

        Ypred = cl.forward(Xtst.to(device)).to(device)
        _, pred = torch.max(Ypred, 1)
        _, labels = torch.max(Ytst , 1)

        correct = float((pred == labels).sum())
        acc = correct / len(Ytst)
        fscore_macro = f1_score(labels, pred, average='macro')

        logging.info('Accuracy:' + str(acc))
        logging.info('Fscore-Marco:' + str(fscore_macro))

    logging.info("Done!")

    return acc, fscore_macro

def test(model, global_weights, Xtst, Ytst):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    params = model.named_parameters()
    dict_params = dict(params)

    i = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.Tensor(global_weights[i]))
            i += 1

    Xtst = np.asarray(Xtst)
    Ytst = np.asarray(Ytst)
    acc, fscore_macro = computeTestErrors(model, Xtst, Ytst, device)

    logging.info('Global Model Testing Done | Accuracy: {:.3%}'.format(acc))
