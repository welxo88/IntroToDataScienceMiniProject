import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import optim, nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter



def classify_gs(dataframe):

    #create log writer
    writer = SummaryWriter('.\\log_files')

    # Process data into usable pandas Series' separating wav-files and their targets
    data = dataframe['data'].values
    data = data.tolist()

    # Replace string valued targets with integer values
    targets = pd.factorize(dataframe['label'])[0]
    # Create a dictionary of the labels for later lookup
    t_labels = dataframe['label'].tolist()
    labels = {}
    for i in range(len(targets)):
        labels[targets[i]] = t_labels[i]
    #print(targets) #numbers
    #print(t_labels) #strings
    #print(labels) #dictionary

    # Split the data to training and test sets
    train_data, test_data, train_target, test_target = train_test_split(data, targets, train_size=0.8, test_size=0.2)
    print("Training data size:", len(train_data))
    print("Test data size:", len(test_data))
    print()
    
    # Transform data arrays to tensors
    train_data = Variable(torch.from_numpy(np.array(train_data))).float()
    train_target = Variable(torch.from_numpy(np.array(train_target))).float()
    test_data = Variable(torch.from_numpy(np.array(test_data))).float()
    test_target = Variable(torch.from_numpy(np.array(test_target))).float()

    n_data = len(data) # Number of data points
    n_feat = 88375     # Number of features
    n_hidden = 15      # Number of nodes in hidden layer
    n_output = 1       # Size of output

    # Initialize weight matrices to normal distribution
    W1 = nn.init.normal_(torch.empty(n_feat, n_hidden))
    b1 = nn.init.normal_(torch.empty(n_hidden, ))
    W2 = nn.init.normal_(torch.empty(n_hidden, n_output))
    b2 = nn.init.normal_(torch.empty(n_output, ))
    weights = [W1, b1, W2, b2]
    
    # Set require_weights to get gradients from PyTorch
    for index, w in enumerate(weights):
        w = Variable(w, requires_grad=True)
        weights[index] = w

    # Initialize optimizer for gradient descent
    lr = 0.01
    opt = optim.SGD(weights, lr=lr)

    # Fit the weight matrix to data with rng iterations
    rng = 1000
    print("Fitting to data \nLearning rate: %.2f\nTraining iterations: %d\nNumber of hidden nodes: %d\n" % (lr, rng, n_hidden))
    for i in range(rng):
        # Initialize gradients to prevent buildup
        opt.zero_grad()
        # Calculate loss of prediction
        train_loss = loss(train_data, train_target, weights)
        
        #log loss & accuracy
        true_pos, false_pos = accuracy(test_data, test_target, weights)
        writer.add_scalars('data', {'train_loss': train_loss, 'true_positive': true_pos, 'false_positive': false_pos}, i)
                
        # Backpropagate: Compute sum of gradients
        train_loss.backward()
        if i == 0:
            print("Training loss on the first iteration: %.8f" % (train_loss.item()))
        elif (i+1) % 20 == 0:
            print("Training loss on the %dth iteration: %.8f" % (i+1, train_loss.item()))
        # Single optimization step
        opt.step()

    # Loss in test_data
    print(loss(test_data, test_target, weights).item())

    # Save weights for later use with the number of hidden nodes to minimize compatibility problems
    f_name = 'weights%d' % n_hidden
    np.save(f_name, np.array(weights))

    #save scalars
    writer.export_scalars_to_json(".\\all_scalars.json")
    writer.close()

    # Return predictions based on test data, test targets, labels for test targets and the weight matrix
    return model(torch.from_numpy(np.array(test_data)).float(), weights), test_target, labels, weights


def loss(x, y, weights):
    """
    :param x: Input vector
    :param y: Correct outputs
    :return: Mean squared error
    """
    y_pred = model(x, weights).squeeze()
    y_pred = (y_pred-y)**2
    return y_pred.sum()/len(y_pred)


def model(x, weights):
    """
    :param x: Input vector
    :param weights: ANN weights
    :return: Output vector of ANN
    """
    W1, b1, W2, b2 = weights
    return torch.mm(torch.sigmoid(torch.mm(x, W1)+b1), W2)+b2

def accuracy(data, target, weights):
    """
    :param data: Input data vector
    :param target: Correct outputs
    :param weights: ANN weights
    :return: True positives, false positives
    """
    pred = np.array(model(data, weights).squeeze().tolist())
    pred_guns = pred < 0.5
    target_guns = np.array(target) < 0.5
    true_positives = pred_guns & target_guns
    print('pred guns %d, target guns %d' % (sum(pred_guns), sum(target_guns)))
    print('tru pos %f, false pos %f' % (sum(true_positives), sum(true_positives)/sum(pred_guns)))
        
    return sum(true_positives)/sum(target_guns), sum(true_positives)/sum(pred_guns)
    


df = pd.read_pickle("dataset.pkl")

test_pred, test_target, labels, weights = classify_gs(df)
test_pred_list = np.array(test_pred.flatten().round_().tolist())
test_target_list = np.array(test_target.tolist())