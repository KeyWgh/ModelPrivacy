'''Common defense methods.'''
import torch


class SimData(torch.utils.data.Dataset):
    """Torch dataset for data loader."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


class Defense:
    """Base defense class for classification task."""
    def __init__(self, dataloader, teacher, device='cpu', tol=1e-10, add=True):
        """
        Parameters
        ----------
        dataloader : torch.utils.data.Dataset
            Dataset loader.
        teacher : function
            Teacher model.
        device : str
            CPU or GPU device used for training.
        tol : float
            Clip the probability to [tol, 1-tol].
        add : bool
            Add the generated perturbation or not.
        """
        self.data = []  # retrieve the predictors of the training sample
        self.target = []  # retrieve the outcome of the teacher model
        self.true_target = []  # retrieve the outcome of the training sample
        teacher.train(False)
        for data, target in dataloader:
            output = teacher(data.to(device))
            self.data.append(data)
            self.target.append(output)
            self.true_target.append(target.to(device))
        self.data = torch.vstack(self.data)
        self.target = torch.vstack(self.target).squeeze().detach()
        self.true_target = torch.hstack(self.true_target).squeeze().detach()
        self.device = device
        self.tol = tol

        if add:
            self.y = self._scale(self._clip(self.add_noise(), self.tol))
        else:
            self.y = self.target

    def add_noise(self):
        """Calculate the perturbation and add it to the responses."""
        return self.target

    def sample(self, n=None, batch_size=128, type='defended'):
        """Sample the dataset.
        Parameters
        ----------
        n : int
            Sample size.
        batch_size : int
            Batch size.
        type : str
            defended : sample the perturbed dataset
            teacher : sample the undefended dataset
            All others : sample the original dataset

        Returns
        -------
        torch.utils.data.DataLoader
        """
        if n is None:
            n = self.data.shape[0]
        if type == 'defended':
            attack_set = SimData(self.data[:n], self.y[:n])
        elif type == 'teacher':
            attack_set = SimData(self.data[:n], self.target[:n])
        else:
            attack_set = SimData(self.data[:n], self.true_target[:n])
        return torch.utils.data.DataLoader(attack_set, batch_size=batch_size, shuffle=True)

    @staticmethod
    def _scale(y):
        """Rescale a vector such that each instance sums to one."""
        return (y.T/y.sum(dim=1)).T if len(y.shape) > 1 else y

    @staticmethod
    def _clip(y, tol):
        """Clip the probability."""
        return torch.clip(y, tol, 1-tol)


class RandomNoise(Defense):
    """Add random Gaussian noise."""
    def __init__(self, sigma=0.1, *args, **kwargs):
        """
        Parameters
        ----------
        sigma : float
            Variance of the Gaussian distribution.
        """
        self.sigma = sigma
        super().__init__(*args, **kwargs)

    def add_noise(self):
        x = self.target
        return x + torch.normal(0, self.sigma, size=x.shape, device=self.device)


class DeceptiveNoise(Defense):
    """Add deceptive noise."""
    def __init__(self, gamma=0.1, beta=1, *args, **kwargs):
        """
        Parameters
        ----------
        sigma : float
            Range of noise.
        beta : float
            Scale of noise.
        """
        self.gamma = gamma
        self.beta = beta
        super().__init__(*args, **kwargs)

    @staticmethod
    def _r(x, gamma):
        return torch.sigmoid(gamma * torch.log(x / (1 - x))) - 1 / 2

    def add_noise(self):
        y = self.target
        return y-self.beta*self._r(y, self.gamma)


class AdaptiveNoise(Defense):
    """Add adaptive noise."""
    def __init__(self, mis_model, tau=0.8, nu=-1000, *args, **kwargs):
        """
        Parameters
        ----------
        mis_model : function
            The mis-specified model that determines noise.
        tau : float
            Threshold for suspicious inputs.
        nu : float
            Scale of noise.
        """
        self.tau = tau
        self.nu = nu
        super().__init__(add=False, *args, **kwargs)
        self.mis_target = mis_model(self.data.to(self.device)).detach()  # Outcomes of the wrong model
        self.y = self._scale(self.add_noise())

    def add_noise(self):
        excess = self.target.max(dim=1).values - self.tau
        alpha = torch.sigmoid(excess * self.nu)
        return (1 - alpha)[:, None] * self.target + alpha[:, None] * self.mis_target


activation = {}


def get_activation(name):
    """Get the outputs from the intermedia layers of a neural network."""
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


class Overfit(Defense):
    """Add higher order polynomial noise for NN."""
    def __init__(self, dataloader, teacher, epsilon=1, device='cpu'):
        # TODO : attempt failed
        self.data = []
        self.target = []
        self.y = []
        self.true_target = []
        teacher.train(False)
        # activation = {}
        teacher.classifier[-2].register_forward_hook(get_activation('last'))
        teacher.classifier[-1].register_forward_hook(get_activation('logits'))
        # dsg_matrix = torch.zeros(84, 10, device=device)
        # dsg_matrix[:, 0] = epsilon
        dsg_matrix = torch.rand(size=(84, 10), device=device) * epsilon
        for data, target in dataloader:
            output = teacher(data.to(device))
            self.data.append(data)
            self.target.append(output)
            tmp = activation['last']
            logits = activation['logits']+torch.pow(tmp, 3)@dsg_matrix
            self.y.append(F.softmax(logits, dim=1))
            self.true_target.append(target.to(device))

        self.data = torch.vstack(self.data)
        self.target = torch.vstack(self.target).detach()
        self.y = torch.vstack(self.y).detach()
        self.y = self._scale(self.y)
        self.true_target = torch.hstack(self.true_target)
