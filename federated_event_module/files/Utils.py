import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import preprocessing
import numpy as np



def label_encoder(df):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df)
    list_class = enc.categories_
    df_array = enc.transform(df).toarray() #Encode the classes to a binary array
    return df_array, list_class

def Label_Encoder(labels):
    le = LabelEncoder()
    le.fit(labels)
    labels_class = le.classes_
    print(labels_class.shape)
    code = le.transform(labels)
    df = pd.DataFrame(code)
    return df, labels_class

def normalize(df):
    x = df #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def data_wrapper(x, y, batch_size=128):

    xtr = torch.FloatTensor(x)
    ytr = torch.FloatTensor(y)

    #Wrap the dataset into the tensor dataset
    dataset = Data.TensorDataset(xtr, ytr)


    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    batch, labels = next(iter(loader))
    print(batch.shape)
    return loader

def train_model(model, epoch, loader, optimizer,loss_function, train_scheduler):
    for e in range(1, epoch + 1):
        e_loss = 0
        train_acc = 0

        for i, record in enumerate(loader):
            (features, labels) = record
            correct = 0
            #labels = labels.cuda()
            #features = features.cuda()

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            _p, tr_pred = torch.max(outputs, 1)
            vp, labels = torch.max(labels, 1)

            #Get training accuracy
            correct+= float((tr_pred == labels).sum())
            ba_accuracy = float(100*(correct/features.shape[0]))
            train_acc += ba_accuracy

            train_scheduler.step(e)
            e_loss += loss.item()


        e_loss = e_loss/len(loader)
        train_accuracy = train_acc/len(loader)

        print('Training Epoch: {epoch} \tLoss: {:0.6f} \tTrain_ac: {:0.6f} \tLR: {:0.6f}'.format(
            e_loss,
            train_accuracy,
            optimizer.param_groups[0]['lr'],
            epoch=e))

    return model

def test_accuracy_multi(model, d_X, d_y):
    correct = 0
    with torch.no_grad():
        confidence_score = model(d_X)
        _, pred = torch.max(confidence_score, 1)
        _, labels = torch.max(d_y, 1)

        correct = float((pred == labels).sum())
        acc = correct / len(d_y)
        return acc, confidence_score, pred
