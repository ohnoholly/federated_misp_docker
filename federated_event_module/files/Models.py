import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class Multi_Classifier_Reg(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, out_dim):
        super(Multi_Classifier_Reg,self).__init__()
        self.fc1 = nn.Linear(in_dim, h1_dim)
        self.bn1 = nn.BatchNorm1d(h1_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.bn2 = nn.BatchNorm1d(h2_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(h2_dim, h3_dim)
        self.bn3 = nn.BatchNorm1d(h3_dim)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.1)
        self.out = nn.Linear(h3_dim, out_dim)
        self.out_act = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        x = self.out(x)
        x = self.out_act(x)
        return x


class Multi_CNN(nn.Module):

    def __init__(self, input_dim, output_dim, sign_size=32, cha_input=16, cha_hidden=32,
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super(Multi_CNN,self).__init__()

        hidden_size = sign_size*cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = dense1

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input,
            cha_input*K,
            kernel_size=5,
            stride = 1,
            padding=2,
            groups=cha_input,
            bias=False)
        self.conv1 = conv1

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = conv2

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = conv3


        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_hidden,
            bias=False)
        self.conv4 = conv4

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = dense2

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x


class DNN(pl.LightningModule):

    def __init__(self, input_dim, output_dim, nn_depth, nn_width, dropout, momentum):
        super().__init__()

        self.bn_in = nn.BatchNorm1d(input_dim, momentum=momentum)
        self.dp_in = nn.Dropout(dropout)
        self.ln_in = nn.Linear(input_dim, nn_width, bias=False)

        self.bnorms = nn.ModuleList([nn.BatchNorm1d(nn_width, momentum=momentum) for i in range(nn_depth-1)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(nn_depth-1)])
        self.linears = nn.ModuleList([nn.Linear(nn_width, nn_width, bias=False) for i in range(nn_depth-1)])

        self.bn_out = nn.BatchNorm1d(nn_width, momentum=momentum)
        self.dp_out = nn.Dropout(dropout/2)
        self.ln_out = nn.Linear(nn_width, output_dim, bias=False)

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.bn_in(x)
        x = self.dp_in(x)
        x = nn.functional.relu(self.ln_in(x))

        for bn_layer,dp_layer,ln_layer in zip(self.bnorms,self.dropouts,self.linears):
            x = bn_layer(x)
            x = dp_layer(x)
            x = ln_layer(x)
            x = nn.functional.relu(x)

        x = self.bn_out(x)
        x = self.dp_out(x)
        x = self.ln_out(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        m = nn.Softmax()
        y_probs = m(y_logit)

        loss = self.loss(y_logit, y)
        _, pred = torch.max(y_probs, 1)
        _, labels = torch.max(y, 1)

        correct = float((pred == labels).sum())
        acc = correct / len(y)
        fscore_micro = f1_score(labels, pred, average='micro')
        fscore_macro = f1_score(labels, pred, average='macro')
        self.log('Test_loss', loss)
        self.log('Accuracy', acc)
        self.log('Fscore-Micro', fscore_micro)
        self.log('Fscore-Marco', fscore_macro)

    def predict_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        m = nn.Softmax()
        y_probs = m(y_logit)
        _, pred = torch.max(y_probs, 1)
        _, labels = torch.max(y, 1)

        return labels, pred

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'valid_loss',
        }
        return [optimizer], [scheduler]




class SoftOrdering1DCNN(pl.LightningModule):

    def __init__(self, input_dim, output_dim, sign_size=32, cha_input=16, cha_hidden=32,
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        hidden_size = sign_size*cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input,
            cha_input*K,
            kernel_size=5,
            stride = 1,
            padding=2,
            groups=cha_input,
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)


        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_hidden,
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        m = nn.Softmax()
        y_probs = m(y_logit)

        loss = self.loss(y_logit, y)
        _, pred = torch.max(y_probs, 1)
        _, labels = torch.max(y, 1)

        correct = float((pred == labels).sum())
        acc = correct / len(y)
        fscore_micro = f1_score(labels, pred, average='micro')
        fscore_macro = f1_score(labels, pred, average='macro')
        self.log('Test_loss', loss)
        self.log('Accuracy', acc)
        self.log('Fscore-Micro', fscore_micro)
        self.log('Fscore-Marco', fscore_macro)


    def predict_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        m = nn.Softmax()
        y_probs = m(y_logit)
        _, pred = torch.max(y_probs, 1)
        _, labels = torch.max(y, 1)

        return labels, pred

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'valid_loss',
        }
        return [optimizer], [scheduler]
