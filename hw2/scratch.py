# assuming x is n x d, 4 x 5
# y is 4 x 1 (column vector, ground truth)
# c is also n x d, with values on the diagonal
# w matches the dimension of xi (an observation from x) - in this case since xi is 1x4, w is 1x4
#           so wTx = 4 x 1 . 1 x 4 = 4 x 4

## NO, THINK LIKE q1! --> well, I think its a case by case basis. when X is n x d(like normal) just swap all the numbers below around:
# x is d x n (ie 5 x 4), 
# xi is a column vector (d x 1) so (5 x 1)
# y is 1 x n (ie 1 x 4) --> a row containing the ground truth for each observation (x has 4 columns, 4 observations, each with 5 features(rows))
# Big C is n x n!!!!!!!
# little c is just a row/column vector of values (doesn't matter since you're not multiplying by little c)
# ci then is just a value
# w matches dimension of xi, since xi is 5 x 1 (a single column of 5 features is an example in this case) w is also 5 x 1
#           so w^Tx is 1 x 5 . 5 x 1 = 1 x 1 (a single value)
# yi is also a single value, since y istelf is a row vector, so y at i is just the ith element of the row
# b is just a single value

c = [10, 20, 30, 40]    # this is an array of scalars - c at i is 10 for ex
xi = [1,2,3,4,5]      # i'm not going to represent them like I'm supposed to but this is supposed to be a column vector - literally x at i
y = [1,0,1,1]   # row vector - treated like an array of scalars - y at i is 1 for ex
w = [.12, .22, .33, .44, .55]   # supposed to be column vector