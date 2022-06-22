from utils import *
import numpy as np
from numpy.linalg import norm, pinv
from collections import defaultdict

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets


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


def gram(grad_list, Y=None):
    """Calculate the gram matrix from the neural network.
    Parameters
    ----------
    grad_list : List
        List of gradients for each layer.
    Y : None or List
        None then same as grad_list.

    Returns
    -------
    numpy.ndarray
    """
    n = len(grad_list)
    m = n if Y is None else len(Y)
    gram = np.zeros((n, m))
    for i in range(n):
        u = grad_list[i]
        for j in range(m):
            if j < i and Y is None:
                gram[i, j] = gram[j, i]
            else:
                v = Y[j] if Y else grad_list[j]
                gram[i, j] = np.sum(list(map(lambda x: torch.sum(x[0]*x[1]).item(), zip(u, v))))
    return gram


def recode(x):
    """Turn MNIST to a binary classification task by choosing label 1 and 7."""
    return 1 if x == 7 else 0
    # return x


class LeNet5(nn.Module):
    """LeNet."""
    def __init__(self, n_classes=1):
        super(LeNet5, self).__init__()

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


def train(model, train_loader, criterion, optimizer):
    """Training for one epoch."""
    train_loss = 0
    model.train(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Initialize grad
        output = model(data.to(device))  # feed model
        loss = criterion(output.squeeze(), target.squeeze().to(device))   # calculate loss
        loss.backward()  # back propagation
        optimizer.step()  # update parameters
        train_loss += loss.item()  # sum up training loss
    return train_loss / len(train_loader)


def test(model, test_loader, criterion):
    """Test error on the test dataset given a trained model."""
    model.train(False)
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += criterion(output.squeeze(), target.squeeze().to(device)).item()  # sum up batch loss
    test_loss = test_loss / len(test_loader)
    return test_loss


def myCustomLoss(my_outputs, my_labels):
    """Mis-classification error."""
    # specifying the batch size
    my_batch_size = my_outputs.size()[0]

    my_outputs = my_outputs > 0
    my_labels = my_labels > 0
    # return the results
    return torch.sum(my_outputs != my_labels)/my_batch_size

# def myCustomLoss(my_outputs, my_labels):
#     '''Misclassification error.'''
#     #specifying the batch size
#     my_batch_size = my_outputs.size()[0]
#
#     my_outputs = my_outputs > 0.5
#     my_labels = my_labels > 0.5
#     #returning the results
#     return torch.sum(my_outputs != my_labels).item()/my_batch_size


def CELoss(my_outputs, targets):
    """Cross entropy loss or negative log likelihood, automatically chosen based on the hard/soft label."""
    if my_outputs.shape == targets.shape:
        return nn.KLDivLoss(reduction='batchmean')(torch.log(my_outputs), targets)
    else:
        return nn.NLLLoss()(torch.log(my_outputs), targets)


def cal_an(m, n, loss, type='teacher'):
    """Evaluate the utility loss."""
    return loss(m.y[:n], m.target[:n]).item() if type == 'teacher' else loss(m.y[:n], m.true_target[:n].to(device)).item()


def run(train_loader, test_loader, model, criterion = CELoss, criterion2 = myCustomLoss, lr=0.001, num_epochs=100, verbose=True):
    """Model training procedure."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5)
    qtl = num_epochs // 10
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer)
        test_loss = test(model, test_loader, criterion2)
        scheduler.step()
        if (epoch % qtl == 0) & verbose:
            print('Train({})[{:.0f}%]: Loss: {:.4f}; Test error:{:.4f}'.format(
                epoch, 100. * epoch / num_epochs, train_loss, test_loss))

    return train_loss, test_loss, model


def teacherMSELoss(my_outputs, targets):
    return nn.MSELoss()(torch.sigmoid(my_outputs), targets.to(torch.double))

# def teacherMSELoss(my_outputs, targets):
#     return nn.MSELoss()(my_outputs, targets.to(torch.double))


# The data is converted to range [0, 1], and select categories 1 and 7 only.
# Change data path accordingly!
transform = torchvision.transforms.ToTensor()
mnist_trainset0 = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
idx = torch.logical_or(mnist_trainset0.targets==1, mnist_trainset0.targets==7)
mnist_trainset0.data = mnist_trainset0.data[idx]
mnist_trainset0.targets = (mnist_trainset0.targets[idx]).apply_(recode)

mnist_testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
idx = torch.logical_or(mnist_testset.targets==1, mnist_testset.targets==7)
mnist_testset.data = mnist_testset.data[idx]
mnist_testset.targets = (mnist_testset.targets[idx]).apply_(recode)

batch_size = 128

torch.set_default_dtype(torch.float64)
device = 'cpu'
nrep = 1
res_nn = []
ans = np.zeros(nrep)
methods = ['W/O', 'RN', 'DP', 'AM', 'Our']
emp = 1000
ntest = 1000
# Compare some defense mechanisms
for i in range(nrep):
    # Split training and test
    mnist_trainset, valid = torch.utils.data.random_split(mnist_trainset0, (5000, 8007))
    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)

    # Teacher model
    train_loss, test_loss, teacher = run(trainloader, testloader, LeNet5(1).to(device),
                                         criterion=teacherMSELoss, num_epochs=50,
                                         verbose=False)

    df = {method: defaultdict(list) for method in methods}
    df['Teacher'] = {'MSE': test(teacher, testloader, teacherMSELoss), '01': test(teacher, testloader, myCustomLoss)}
    n = 100

    # Query data
    valid2 = torch.utils.data.Subset(mnist_trainset0, valid.indices[:3000])
    validloader = torch.utils.data.DataLoader(valid2, batch_size=batch_size, shuffle=True)
    mt = Defense(testloader, teacher, device=device, add=False)
    ttloader = torch.utils.data.DataLoader(SimData(mt.data, mt.target),
                                           batch_size=batch_size, shuffle=True)

    m = Defense(validloader, teacher, device=device, add=False)
    trainloader = m.sample(n)

    data = m.data
    target = m.target
    training_data = data[:n]
    training_target = target[:n]

    net = LeNet5(1)
    grad_list = []
    for idx in range(n):
        val = net(training_data[idx][None, :])
        grad_list.append(torch.autograd.grad(val, net.parameters(), create_graph=True))
    ker = gram(grad_list)

    emp_grad_list = []
    for idx in range(emp):
        val = net(torch.Tensor(data[n+idx][None, :]))
        emp_grad_list.append(torch.autograd.grad(val, net.parameters(), create_graph=True))

    emp_ker = gram(grad_list, emp_grad_list)
    alpha = 0.0
    ik = pinv(ker+alpha*np.identity(n))
    eker = emp_ker @ emp_ker.T / emp
    mker = ik.T @ eker @ ik
    ey = emp_ker @ target[n:n+emp].numpy() / emp
    u = training_target.numpy() @ mker - ik @ ey

    new_grad_list = []
    for _ in range(ntest):
        val = net(torch.Tensor(mt.data[_]))
        new_grad_list.append(torch.autograd.grad(val, net.parameters(), create_graph=True))

    pred_ker = gram(grad_list, new_grad_list)

    # No noise
    pred_nn = training_target@ik@pred_ker
    df['W/O']['CE_an_teacher'].append(cal_an(m, n, nn.MSELoss()))
    df['W/O']['01_an_teacher'].append(cal_an(m, n, myCustomLoss))
    df['W/O']['CE_an_origin'].append(cal_an(mt, n, teacherMSELoss, 'origin'))
    df['W/O']['01_an_origin'].append(cal_an(mt, n, myCustomLoss, 'origin'))
    df['W/O']['CE_bn_origin'].append(teacherMSELoss(pred_nn, mt.true_target[:ntest]).item())
    df['W/O']['01_bn_origin'].append(myCustomLoss(pred_nn, mt.true_target[:ntest]).item())
    df['W/O']['CE_bn_teacher'].append(nn.MSELoss()(pred_nn, mt.target[:ntest]).item())
    df['W/O']['01_bn_teacher'].append(myCustomLoss(pred_nn, mt.target[:ntest]).item())

    # Random Gaussian noise
    sigma = 0.4
    training_noise_target = training_target+torch.normal(0, sigma, size=training_target.shape, device=device)
    an = nn.MSELoss()(training_noise_target, training_target).item()
    ans[i] = an
    pred_nn = training_noise_target @ ik @ pred_ker
    method = 'RN'
    df[method] = defaultdict(list)
    df[method]['CE_an_teacher'].append(an)
    df[method]['01_an_teacher'].append(myCustomLoss(training_noise_target, m.target[:n]).item())
    df[method]['CE_an_origin'].append(teacherMSELoss(training_noise_target, m.true_target[:n]).item())
    df[method]['01_an_origin'].append(myCustomLoss(training_noise_target, m.true_target[:n]).item())
    df[method]['CE_bn_origin'].append(teacherMSELoss(pred_nn, mt.true_target[:ntest]).item())
    df[method]['01_bn_origin'].append(myCustomLoss(pred_nn, mt.true_target[:ntest]).item())
    df[method]['CE_bn_teacher'].append(nn.MSELoss()(pred_nn, mt.target[:ntest]).item())
    df[method]['01_bn_teacher'].append(myCustomLoss(pred_nn, mt.target[:ntest]).item())

    # Proposed best perturbation
    e = sol(mker, 2 * u)
    e = np.sqrt(n * an) * e / norm(e)
    training_noise_target = training_target + torch.Tensor(e)
    pred_nn = training_noise_target @ ik @ pred_ker
    method = 'Our'
    df[method] = defaultdict(list)
    df[method]['CE_an_teacher'].append(an)
    df[method]['01_an_teacher'].append(myCustomLoss(training_noise_target, m.target[:n]).item())
    df[method]['CE_an_origin'].append(teacherMSELoss(training_noise_target, m.true_target[:n]).item())
    df[method]['01_an_origin'].append(myCustomLoss(training_noise_target, m.true_target[:n]).item())
    df[method]['CE_bn_origin'].append(teacherMSELoss(pred_nn, mt.true_target[:ntest]).item())
    df[method]['01_bn_origin'].append(myCustomLoss(pred_nn, mt.true_target[:ntest]).item())
    df[method]['CE_bn_teacher'].append(nn.MSELoss()(pred_nn, mt.target[:ntest]).item())
    df[method]['01_bn_teacher'].append(myCustomLoss(pred_nn, mt.target[:ntest]).item())

    res_nn.append(df)
