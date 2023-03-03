"""Example of neural networks on MNIST."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import logging
logger = logging.getLogger(__name__)


class SimData(torch.utils.data.Dataset):
    """Torch dataset for data loader."""
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


class LinearNeuralTangentKernel(nn.Linear):
    """Network for neural tangent kernel."""
    def __init__(self, in_features, out_features, bias=True, beta=0.5, w_sig=1):
        self.beta = beta
        super(LinearNeuralTangentKernel, self).__init__(in_features, out_features, bias=bias)
        self.reset_parameters()
        self.w_sig = w_sig

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input):
        return F.linear(input, self.w_sig * self.weight / np.sqrt(self.in_features), self.beta * self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta
        )


class FCNTK(nn.Module):
    """The neural network $f(x)=\sum a_i \sigma(w_i^Tx)$."""
    def __init__(self, *dims):
        super(FCNTK, self).__init__()
        self.layers = nn.ModuleList(
            [LinearNeuralTangentKernel(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for theta in self.layers:
            x = theta(x)
            if theta is not self.layers[-1]:
                x = F.relu(x)
        return x


class LeNet5NTK(nn.Module):
    """LeNet."""
    def __init__(self, n_classes=1):
        super(LeNet5NTK, self).__init__()

        self.feature_extractor = nn.Sequential(
            LinearNeuralTangentKernel(in_features=28 * 28, out_features=200),
            nn.ReLU(),
            LinearNeuralTangentKernel(in_features=200, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x.view(-1, 28 * 28))
        # return torch.sigmoid(x)
        # return the logit value
        return x


class LeNet5(nn.Module):
    # require 32*32 pixels
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return probs


class FC(nn.Module):
    """The fully connected neural network"""

    def __init__(self, *dims):
        super(FC, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for theta in self.layers:
            x = theta(x)
            if theta is not self.layers[-1]:
                x = F.relu(x)
        return x


def train(model, train_loader, criterion, optimizer, device):
    """Training for one epoch."""
    train_loss = 0
    model.train(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Initialize grad
        output = model(data.to(device))  # feed model
        loss = criterion(output.squeeze(), target.to(device).squeeze())   # calculate loss
        loss.backward()  # back propagation
        optimizer.step()  # update parameters
        train_loss += loss.item()  # sum up training loss
    return train_loss / len(train_loader)


def test(model, test_loader, criterion, device):
    """Test error on the test dataset given a trained model."""
    model.train(False)
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += criterion(output.squeeze(), target.to(device).squeeze()).item()  # sum up batch loss
    test_loss = test_loss / len(test_loader)
    return test_loss


def zero_one_loss_binary(my_outputs, my_labels, threshold=0.5):
    """Mis-classification error."""
    # Y should be a tensor of size (batch_size, 1)
    # specifying the batch size
    my_batch_size = my_outputs.size()[0]
    my_outputs = my_outputs > threshold
    my_labels = my_labels > threshold
    # return the results
    return torch.sum(my_outputs != my_labels)/my_batch_size


def zero_one_loss(my_outputs: torch.Tensor, my_labels: torch.Tensor):
    """Mis-classification error."""
    my_batch_size = my_outputs.size()[0]
    my_outputs = my_outputs.argmax(dim=1)
    my_labels = my_labels.argmax(dim=1) if my_labels.ndim > 1 else my_labels
    return torch.sum(my_outputs != my_labels)/my_batch_size


def CELoss(my_outputs, targets, logit=True):
    """Cross entropy loss or negative log likelihood, automatically chosen based on the hard/soft label."""
    # logit: True when my_outputs is predicted probability
    if logit:
        my_outputs = torch.log(my_outputs)
    if my_outputs.shape == targets.shape:
        return nn.KLDivLoss(reduction='batchmean')(my_outputs, targets)
    else:
        return nn.NLLLoss()(my_outputs, targets)


def cal_an(y_true, yhat, loss):
    """Evaluate the utility loss."""
    return loss(y_true, yhat).item()


def run(train_loader, model, test_loader=None, criterion=CELoss, criterion2=zero_one_loss, lr=0.001, num_epochs=100,
        device='cpu', scheduler=None, optimizer=None):
    """Model training procedure."""
    logger.debug(f'Training start with Learning rate: {lr}; Epochs: {num_epochs}; Device: {device}.')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optimizer is None else optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    scheduler = scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5)
    model.to(device)
    qtl = max(num_epochs // 10, 1)
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device=device)
        test_loss = test(model, test_loader, criterion2, device=device) if test_loader is not None else np.nan
        if scheduler:
            scheduler.step()
        if epoch % qtl == 0:
            logger.debug('Train({})[{:.0f}%]: Loss: {:.4f}; Test error:{:.4f}'.format(
                epoch, 100. * epoch / num_epochs, train_loss, test_loss))

    return train_loss, test_loss, model


def teacherMSELoss(my_outputs, targets):
    return nn.MSELoss()(torch.sigmoid(my_outputs), targets.to(torch.double))

# def teacherMSELoss(my_outputs, targets):
#     return nn.MSELoss()(my_outputs, targets.to(torch.double))


def recode(x, cat=1):
    """Turn MNIST to a binary classification task by choosing label 1 and 7."""
    return 1 if x == cat else 0
    # return x


if __name__ == '__main__':
    pass
