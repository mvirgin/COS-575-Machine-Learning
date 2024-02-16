### Matthew Virgin
### COS 575
### Homework 1
### Due 16 February 2024 5 pm

import numpy as np

## Bayesian spam filtering

z = np.genfromtxt('spambase.data', dtype=float, delimiter=',')
np.random.seed(0) #Seed the random number generator
rp = np.random.permutation(z.shape[0]) #random permutation of indices
z = z[rp,:] #shuffle the rows of z
x = z[:,:-1]    # each row without the last colum - last column is target
y = z[:,-1]     # each rows last column

## each column of x represents an input variable, with each row representing
## 1 email. Grab the median of each column(variable) to quantize all of the 
## values of x to either 1 or 2
medians = np.median(x,axis=0)   # 57 medians for 57 variables

## Quantize x
## iterate over columns of x, changing each value in the column to 1 or 2
## depending on if value is above or below median
for i in range(x.shape[1]):         
    ## <= means exactly the median = 1, < means exactly the median = 2
    x[:,i] = np.where(x[:,i] <= medians[i], 1, 2)   # <= gave better results!

## get 2,000 training samples, rest are test
x_train = x[:2000]
y_train = y[:2000]

x_test = x[2000:]
y_test = y[2000:]

def train_and_test(x_train, x_test, y_train, y_test):
    ## calculate prior probability of class 0
    pi_hat_0 = len(y_train[y_train==0]) / len(y_train)

    ## calculate prior probability of class 1 (same as 1-pi_hat_0)
    pi_hat_1 = len(y_train[y_train==1]) / len(y_train)

    likelihood_X1_Y0 = []      # compute likelihoods (conditional probabilities)
    likelihood_X2_Y0 = []      # to use in posterior calculation later

    likelihood_X1_Y1 = []
    likelihood_X2_Y1 = []

    ## loop over columns of train set
    for i in range(x_train.shape[1]):       # i represents column #
        count_instance_X1_Y0 = 0
        count_y0 = 0
        count_instance_X1_Y1 = 0
        count_y1 = 0
        column = x_train[:,i]
        ## calculate probability of 1 occuring in this column given y is 0
        for j in range(len(column)): # loop over the elements of the column
            if column[j] == 1 and y[j] == 0: # get instances where x 1 given y 0
                count_instance_X1_Y0 = count_instance_X1_Y0 + 1
                ## 1 - this is prob x 2 given y 0
                count_y0 = count_y0 + 1
            elif column[j] == 1 and y[j] == 1:  # x 1 given y 1
                count_instance_X1_Y1 = count_instance_X1_Y1 + 1
                ## 1 - this is prob x 2 given y 1
                count_y1 = count_y1 + 1
            elif y[j] == 0:
                count_y0 = count_y0 + 1
            elif y[j] == 1:
                count_y1 = count_y1 + 1
        ## probabilities for column i
        prob1 = count_instance_X1_Y0 / count_y0    
        prob2 = count_instance_X1_Y1 / count_y1
        likelihood_X1_Y0.append(prob1)
        likelihood_X2_Y0.append(1-prob1)
        likelihood_X1_Y1.append(prob2)
        likelihood_X2_Y1.append(1-prob2)

    y_hat = []

    ## test
    ## determine posteriors for each row of the data, highest one is the pred
    ## for that row
    for i in range(len(x_test)):        # loop over test set rows
        posterior0 = pi_hat_0       # for class 0
        posterior1 = pi_hat_1       # for class 1
        row = x_test[i]
        for j in range(len(row)):
            if row[j] == 1:         # calculate posterior 0 and 1 for each row
                posterior0 = posterior0 * likelihood_X1_Y0[j]
                posterior1 = posterior1 * likelihood_X1_Y1[j]
            else:
                posterior0 = posterior0 * likelihood_X2_Y0[j]
                posterior1 = posterior1 * likelihood_X2_Y1[j]
                
        ## determine which posterior is highest and predict for that row
        if posterior1 > posterior0:
            y_hat.append(1)
        else:
            y_hat.append(0)

    # calculate mismatches between y_hat and y_test
    corrects = np.sum(y_test == y_hat)
    acc = corrects / len(y_test)
    print("Accuracy:", acc, "\nError:", 1-acc)

    ## For sanity check:
    ## calculate prior probability of class 0 for test
    te_pi_hat_0 = len(y_test[y_test==0]) / len(y_test)

    ## calculate prior probability of class 1 for test
    te_pi_hat_1 = len(y_test[y_test==1]) / len(y_test)

    if pi_hat_0 > pi_hat_1:      
        ## always predict majority means all instances of minority are misclassified
        ## so percentage of minority class in dataset is the test error
        print("Test error if we always predicted the majority class from training data:", te_pi_hat_1)
    else:
        print("Test error if we always predicted the majority class from training data:", te_pi_hat_0)

train_and_test(x_train, x_test, y_train, y_test)

## Note: could probably have used numpy matrix operations to skip the need for 
##       nested for loops