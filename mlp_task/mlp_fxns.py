import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from dataclasses import dataclass

from censor_methods import noising, omit_sensitive_data

@dataclass
class Config:
    Din: int = 50 # dim of features
    hidden_dim: int = 64
    batchsize: int = 32
    datasize: int = 6400
    split: float = 0.1 # 10/10/80 test val train
    epochs: int = 60
    lr: float = 0.001
    patience: int = 5
    min_delta: float = 1e-4 # for early stopping

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.Generator().manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        # fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self,x):
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)
        return h.squeeze()
    
def weights_init_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def generatedata(Din, hidden_dim=64, num_samples=500):
    # generate data points using MLP with the dimensionality of Din
    gen_model = MLP(Din,hidden_dim)
    gen_model.apply(weights_init_uniform)
    gen_model.eval()
    with torch.no_grad():
        features = torch.torch.distributions.Uniform(
            low=-2, high=2).sample((num_samples,Din)
        )
        labels = gen_model(features) + torch.normal(0., 0.1, size=(1,num_samples)).squeeze()
    return features, labels

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, verbose=True):
        # look at validation loss to check whether it isn't improving for few steps
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if verbose:
                    print("Early stopping. No improvement in validation loss.")
                return True
        return False
    
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
def train(
    model,
    train_data,
    val_data,
    model_config,
    verbose = True,
):
    train_features, train_labels = train_data
    val_features, val_labels = val_data
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr)
    early_stopper = EarlyStopper(
        patience=model_config.patience, 
        min_delta=model_config.min_delta,
    )
    rmse = RMSELoss()

    losses = []
    val_losses = []
    patience = 0
    for epoch in range(model_config.epochs):
        model.train()
        trainsize = len(train_features)
        indices = torch.randperm(trainsize)
        total_loss = 0
        for batch_start in range(0,trainsize, model_config.batchsize):
            batch_idxs = indices[batch_start: batch_start + model_config.batchsize]
            batch_features = train_features[batch_idxs]
            batch_labels = train_labels[batch_idxs]

            pred = model(batch_features)
            loss = rmse(pred,batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (trainsize // model_config.batchsize)
        losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_features)
            val_loss = rmse(val_pred,val_labels)
        val_losses.append(val_loss.item())

        if epoch % 1 == 0 and verbose:
            print(f"Epoch [{epoch+1}/{model_config.epochs}], "
                f"Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if early_stopper.early_stop(val_loss, verbose):
            break
    return losses, val_losses

def local_loss(y,yhat, min, max):
    rmse = RMSELoss()
    mask = (y >= min) & (y <= max)
    local_y = y[mask]
    local_yhat = yhat[mask]
    local_rmse = rmse(local_yhat, local_y)
    return local_rmse.item()

def data_split(data, val_split=0.1, test_split=0.1, random_state=None):
    # split into train, val, and test sets
    if random_state is not None:
        np.random.seed(random_state)

    # Combine the features and labels into a single array
    features, labels = data
    data = np.column_stack((features, labels))
    np.random.shuffle(data)
    test_len = int(len(data) * test_split)
    val_len = int(len(data) * val_split)
    test_data = data[:test_len]
    val_data = data[test_len:test_len + val_len]
    train_data = data[test_len + val_len:]

    # Split the data back into features and labels
    train_features, train_labels = torch.from_numpy(train_data[:, :-1]), torch.from_numpy(train_data[:, -1])
    val_features, val_labels = torch.from_numpy(val_data[:, :-1]), torch.from_numpy(val_data[:, -1])
    test_features, test_labels = torch.from_numpy(test_data[:, :-1]), torch.from_numpy(test_data[:, -1])
    return (train_features, train_labels), (val_features, val_labels), (test_features, test_labels)

def compute_threshold_from_split(labels, split, region):
    """
    Computes a threshold from a given split ratio based on labels and 
    desired sensitive region.
    """
    sorted_labels = np.sort(labels)
    if region == 'above': 
        index = int(len(sorted_labels) * (1-split)) # top split as sensitive
    elif region == 'below':
        index = int(len(sorted_labels) * split) # bottom split as sensitive
    else:
        raise ValueError("Invalid region. Must be 'above' or 'below'.")
    threshold = sorted_labels[index]
    return threshold

def mlptask_wrapper_v1( # without control over omission probability
        seed, 
        tasks, 
        sensitive_threshold, 
        sensitive_region, 
        verbose=False, 
        sanitycheckplot=False
    ):
    """
    This wrapper function executes a series of training tasks on separate MLP models.
    'tasks' is input list of tuples containing task configurations, each consisting of:
        task_name (string): name of the task
        xnoise (float): gaussian noise level applied to features in sensitive region
        ynoise (float): gaussian noise level applied to labels in sensitive region
        omit (boolean): to omit the sensitive region

    # example of tasks
    tasks = [
        ('baseline', 0, 0, False),
        ('omission', 0, 0, True),
        ('x noise', 2.0, 0, False),
        ('y noise', 0, 1.0, False),
    ]

    Returns:
        results (dictionary): Contains dictionaries of training losses, 
            validation losses, predictions, and y values of the test set, 
            test errors for non-sensitive and sensitive regions.
    """

    # generate data - use a different random seed
    setseed(seed + 1)
    data = generatedata(config.Din, config.hidden_dim, config.datasize)
    features, labels = data
    traindata, valdata, testdata = data_split(
        data, val_split=config.split, test_split=config.split, random_state=seed
    )
    x_test, y_test = testdata
    y_min = y_test.min().item()
    y_max = y_test.max().item()

    all_losses = {}
    all_val_losses = {}
    all_preds = {}
    overall_error = {}
    lower_error = {}
    upper_error = {}
    x_levels = {}
    y_levels = {}
    omission = {}
    # #results = {}
    for task_name, xnoise, ynoise, omit in tasks:
        setseed(seed)
        model = MLP(config.Din, config.hidden_dim)
        task_traindata = traindata
        task_valdata = valdata

        if omit:
            # task_traindata = filterdata_by_label(traindata, threshold=sensitive_threshold, omitregion=sensitive_region)
            # task_valdata = filterdata_by_label(valdata, threshold=sensitive_threshold, omitregion=sensitive_region)
            task_traindata = omit_sensitive_data(
                traindata, 
                threshold=sensitive_threshold, 
                sensitive_region=sensitive_region, 
                omit_probability=1, 
                plot=sanitycheckplot
            )
            task_valdata = omit_sensitive_data(
                valdata, 
                threshold=sensitive_threshold, 
                sensitive_region=sensitive_region, 
                omit_probability=1, 
                plot=False
            )

        elif xnoise or ynoise:
            task_traindata = noising(
                traindata, xnoise, ynoise, sensitive_threshold, plot=sanitycheckplot
            )
            task_valdata = noising(
                valdata, xnoise, ynoise, sensitive_threshold, plot=False
            )

        x_levels[task_name] = xnoise
        y_levels[task_name] = ynoise
        omission[task_name] = omit

        # # losses, val_losses = train(model, task_traindata, task_valdata, verbose=verbose)
        # # model.eval()
        # # with torch.no_grad():
        # #     preds = model(x_test)

        # # # calculate test errors for labelled regions above/below the 'sensitive' threshold
        # # lower_error = local_loss(y_test, preds, y_min, sensitive_threshold) # below sensitive threshold
        # # upper_error = local_loss(y_test, preds, sensitive_threshold, y_max) # above

        # # results[task_name] = {
        # #     'x_noise_level': xnoise,
        # #     'y_noise_level': ynoise,
        # #     'omit': omit,
        # #     'train_loss': losses,
        # #     'val_loss': val_losses,
        # #     'pred': preds,
        # #     'y_test': y_test,
        # #     'lower_error': lower_error,
        # #     'upper_error': upper_error,
        # # }

        all_losses[task_name], all_val_losses[task_name] = train(
            model, task_traindata, task_valdata, verbose=verbose
        )
        model.eval()
        with torch.no_grad():
            preds = model(x_test)
        all_preds[task_name] = preds
        rmse = RMSELoss()
        overall_error[task_name] = rmse(preds, y_test).item()

        # calculate test errors for non-sensitive and sensitive regions
        lower_error[task_name] = local_loss(y_test, preds, y_min, sensitive_threshold)
        upper_error[task_name] = local_loss(y_test, preds, sensitive_threshold, y_max)

    results = {
        'x_noise_level': x_levels,
        'y_noise_level': y_levels,
        'omit': omission,
        'train_loss': all_losses,
        'val_loss': all_val_losses,
        'pred': all_preds,
        'y_test': y_test,
        'overall_error': overall_error,
        'lower_error': lower_error,
        'upper_error': upper_error,
    }
    return results

def mlptask_wrapper_v2( # with control over omission probability
        seed, 
        tasks,  
        censor_region, 
        censor_split=None,
        censor_threshold=None,
        model_config=None,
        verbose=False, 
        sanitycheckplot=False,
    ):
    """
    This wrapper function executes a series of training tasks on separate MLP models.
    'tasks' = list of tuples containing task configs, where each tuple consists of
        task_name (string): name of the task
        xnoise (float): gaussian noise level applied to features in sensitive region
        ynoise (float): gaussian noise level applied to labels in sensitive region
        omit (boolean): to omit the sensitive region
        omit_fraction (float): to set approximate percent of sensitive region being omitted (beteen 0 and 1)

    Sensitive data settings: 
        Provide either sensitive_split OR sensitive_threshold
        sensitive_split (float): between 0 and 1, set the percentage of data points 
            that are considered 'sensitive'.
        sensitive_threshold (float): set threshold value of y to separate sensitive 
            and non-sensitive regions of the data
        sensitive_region(string): specify which region is sensitive, 'above' or 'below' 
            the threshold/split

    Returns:
        results (dictionary): Contains dictionaries of training losses, validation losses, predictions, and y values of
            the test set, test errors for below and above 'sensitive' threshold.
    """
    if model_config == None:
        model_config = Config()
    # generate data
    setseed(seed + 1)
    data = generatedata(model_config.Din, model_config.hidden_dim, model_config.datasize)
    features, labels = data

    # compute threshold from sensitive/non-sensitive split
    if censor_split is not None:
        censor_threshold = compute_threshold_from_split(
            labels, censor_split, censor_region
        )
        print(f"With {censor_split:.0%} 'censored' split,"
              f"censor threshold = {censor_threshold}")
    traindata, valdata, testdata = data_split(
        data, val_split=model_config.split, test_split=model_config.split, random_state=seed
    )
    x_test, y_test = testdata
    y_min = y_test.min().item()
    y_max = y_test.max().item()

    all_losses = {}
    all_val_losses = {}
    all_preds = {}
    overall_error = {}
    lower_error = {}
    upper_error = {}
    x_levels = {}
    y_levels = {}
    omission = {}
    omit_frac = {}
    for task_name, xnoise, ynoise, omit, omit_fraction in tasks:
        setseed(seed)
        model = MLP(model_config.Din, model_config.hidden_dim)
        task_traindata = traindata
        task_valdata = valdata

        if omit:
            task_traindata = omit_sensitive_data(
                traindata,
                threshold=censor_threshold,
                omit_region=censor_region,
                omit_probability=omit_fraction,
                plot=sanitycheckplot
            )
            task_valdata = omit_sensitive_data(
                valdata,
                threshold=censor_threshold,
                sensitive_region=censor_region,
                omit_probability=omit_fraction,
                plot=False
            )

        elif xnoise or ynoise:
            task_traindata = noising(traindata, xnoise, ynoise, censor_threshold, plot=sanitycheckplot)
            task_valdata = noising(valdata, xnoise, ynoise, censor_threshold, plot=False)

        x_levels[task_name] = xnoise
        y_levels[task_name] = ynoise
        omission[task_name] = omit
        omit_frac[task_name] = omit_fraction

        all_losses[task_name], all_val_losses[task_name] = train(
            model, task_traindata, task_valdata, model_config, verbose=verbose
        )
        model.eval()
        with torch.no_grad():
            preds = model(x_test)
        all_preds[task_name] = preds
        rmse = RMSELoss()
        overall_error[task_name] = rmse(preds, y_test).item()

        # calculate test errors for regions above/below the 'sensitive' threshold
        lower_error[task_name] = local_loss(y_test, preds, y_min, censor_threshold)
        upper_error[task_name] = local_loss(y_test, preds, censor_threshold, y_max)

    results = {
        'censor_threshold': censor_threshold,
        'x_noise_level': x_levels,
        'y_noise_level': y_levels,
        'omit': omission,
        'omit_fraction': omit_frac,
        'train_loss': all_losses,
        'val_loss': all_val_losses,
        'pred': all_preds,
        'y_test': y_test,
        'overall_error': overall_error,
        'lower_error': lower_error,
        'upper_error': upper_error,
    }
    return results
