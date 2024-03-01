### Matthew Virgin
### COS 575
### Homework 1
### Due 1 March 2024 5 pm

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

## function to print ith image from dataset x with features d
def print_image(i, x, d):
    plt.imshow( np.reshape(x[:,i], (int(np.sqrt(d)),int(np.sqrt(d)))))
    plt.show()

## takes an array of indices ids, a dataset x with features d, and 
## ground truth y to display images with their true label
def print_images(ids, x, d, y, y_pred):
    num_rows = 4
    num_cols = 5
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))

    for i, ax in enumerate(axes.flat):
        if i < len(ids):
            ax.imshow(np.reshape(x[:, ids[i]], (int(np.sqrt(d)), int(np.sqrt(d)))), cmap='gray')
            ax.set_title(f'i: {ids[i]} True: {y[ids[i]]} Pred: {y_pred[ids[i]]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

## compute sigmoid(z)
## w/ newton method, z will be (w^T)x+b which is (theta^T)(x_aug)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

## compute negative log likelihood as loss function
def loss(x, y, theta, lambda_):
    z = np.dot(theta.T, x)
    y_probs = sigmoid(z)
    reg_term = lambda_ * (np.sum(np.square(theta)))
    l = np.dot(y, np.log(y_probs).T) + np.dot(1-y, np.log(1-y_probs).T)
    ## divide by # of observations to avoid size of dataset affecting it
    l = (l + reg_term)/x.shape[1]   
    return l

## calculating the gradient of negative log likelihood, vectorized approach
## gradient should be d+1 x 1 where d is # of features (784+1)
def gradient(x_train, y_train, theta, lambda_):
    z = np.dot(theta.T, x_train)
    y_probs = sigmoid(z)
    ## remember L2 norm of theta is the same as lambda*(theta^T . theta)
    ## so 1st gradient of that is 2*lambda*theta
    reg_term = 2*lambda_*theta
    grad = np.dot(x_train, (y_probs-y_train).T) + reg_term
    return grad

## calculating hessian, hessian should be d+1 x d+1
def hessian(x_train, theta, lambda_):
    z = np.dot(theta.T, x_train)
    y_preds = sigmoid(z)
    diagonal = np.multiply(y_preds, 1-y_preds)  # element-wise mult
    diag_matrix = np.diag(diagonal[0])
    x_dot_diag = np.dot(x_train, diag_matrix)
    hess = np.dot(x_dot_diag, x_train.T)
    ## reg term 2nd gradient (hessian): 2 * lambda * I (matching shape of hess)
    reg_term = 2*lambda_*np.identity(hess.shape[0])
    hess = hess + reg_term
    return hess

## calculates newton's method using gradient and hessian
## stops at max_its iterations
def newton(x_train, y_train, theta, lambda_, max_its):
    for i in range(max_its):
        g = gradient(x_train, y_train, theta, lambda_)
        h = hessian(x_train, theta, lambda_)
        h_inverse = np.linalg.inv(h)
        theta = theta - np.dot(h_inverse, g)
    return theta

def main():
    ## fetch data
    mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')
    x = mnist_49_3000['x']
    y = mnist_49_3000['y']
    d,n= x.shape

    ## convert ground truth from -1 for '4' and 1 for '9' to 0 for '4', 1 for '9'
    ## to make sigmoid make sense  
    y[y<0] = 0

    ## x is 784 rows by 3000 columns, each column is an image (example)
    ## rows represent features in this case
    ## y is 1 row and 3000 columns, where each column stores image class

    ## Split into train and test
    x_train = x[:, :2000]   # grabs first 2,000 columns of x
    y_train = y[:, :2000]

    x_test = x[:, 2000:]    # grabs all columns after column 2,000 (all remaining)
    y_test = y[:, 2000:]

    # ## augment x to have bias "feature"
    x_train_aug = np.concatenate((x_train, np.ones((1, x_train.shape[1]))), axis=0)
    x_test_aug = np.concatenate((x_test, np.ones((1, x_test.shape[1]))), axis=0)

    ## create theta with bias term
    theta = np.zeros((x_train.shape[0]+1, 1))   # +1 for bias (=shape[0] of x aug)

    ## define lambda
    lambda_ = 10

    ## find optimal theta using training data
    max_its = 100
    theta = newton(x_train_aug, y_train, theta, lambda_, max_its)

    ## find probabilities on test set using optimal theta
    z = np.dot(theta.T, x_test_aug)
    y_probs = sigmoid(z) # row of 1,000 probabilities, 1 per observation in test set

    ## if the probability is greater than 0.5, predict class 1, otherwise 0
    y_pred = []
    for i in range(len(y_probs[0])):
        if y_probs[0][i] >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

    corrects = np.sum(y_test[0] == y_pred)
    acc = corrects / len(y_test[0])
    print("Accuracy:", acc, "\nError:", 1-acc)
    print("Termination criteria was", max_its, "iterations" )

    ## print optimized loss from using best theta
    print("J(theta) for test set at optimal theta:", 
          loss(x_test_aug, y_test, theta, lambda_))
    
    ## find the misclassified images indices
    missed = np.where(y_test[0] != y_pred)[0] # indices will be between 0 and 1k

    ## which of those had a probability above close to 100 or close to 0?
    h_confidence = .7
    l_confidence = .3
    confident_pred_indices = []
    count = 0
    for i in range(len(missed)):
        j = missed[i]   # missed holds misclassified indices
        if y_probs[0][j] >= h_confidence:
            confident_pred_indices.append(j)   
            count = count + 1 # update count
        elif y_probs[0][j] <= l_confidence:
            confident_pred_indices.append(j)   
            count = count + 1
        if count >=20:
            break
    print("my measure of confidence for the misclassified images is merely the" 
            f" probability from sigmoid. If a probability is below {l_confidence},"  
            " the model is confident that image is a 4. If a probability is above" 
            f" {h_confidence}, the model is confident that image is a 9. " 
            "20 missclasifieds out of this bunch is displayed")
    ## display images
    print_images(confident_pred_indices, x_test, d, y_test[0], y_pred) 

main()