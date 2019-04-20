# Exercise: Logistic Regression
``` 
The goal of this exercice is to implement the logistic regression, using loops and vectorization. The main functions to code are the cost function and the gradient in both versions, and the stochastic gradient ascent. <br>
    a) Create a function prediction_function and its vectorized version prediction_function_vec that return a predicted value 
$\hat y ∈{[0, 1]}$ for a given X ∈ R m×n and θ ∈ R n.
Compare these two functions in terms of returned values and computational time. <br>
    b) Create a function gradient and its vectorized version gradient_vec that return a vector of partial derivatives,
the gradient, for a given X ∈ R m×n, θ ∈ R n and y ∈ {0, 1}.
Compare these two functions in terms of returned values and computational time. <br>
    c) Implement your own version of the stochastic gradient ascent.
	What should you do to propose your own version of the batch gradient descent ?<br>
    d) After predicting the y-value for each test example, compute the misclassification error $(\hat e)$ as follows: <br>
$$\hat e = 1/m \sum_{i=1}|\hat y{(i)} − y^{(i)}|$$
where $ \hat y (i)$ is the round value of $hθ(x(i))$
```

import random
filename = "mnist_x_train"
n = sum(1 for line in open("data/mnist/mnist_x_train")) - 1 #number of records in file (excludes header)
s = 500 #desired sample size
skip = sorted(random.sample(range(1, n),n-s)) #the 0-indexed header will not be included in the skip list
trainX = np.loadtxt("data/mnist/mnist_x_train")

filename = "mnist_y_train"
n = sum(1 for line in open("data/mnist/mnist_y_train")) - 1 
s = 500 #desired sample size
skip = sorted(random.sample(range(1, n),n-s))
trainY = np.loadtxt("data/mnist/mnist_y_train")

filename = "mnist_x_test"
n = sum(1 for line in open("data/mnist/mnist_x_test")) - 1 
s = 500 #desired sample size
skip = sorted(random.sample(range(1, n),n-s)) 
testX = np.loadtxt("data/mnist/mnist_x_test")

filename = "mnist_y_test"
n = sum(1 for line in open("data/mnist/mnist_y_test")) - 1 
s = 500 #desired sample size
skip = sorted(random.sample(range(1, n),n-s))
testY = np.loadtxt("data/mnist/mnist_y_test")


trainX.shape, testX.shape, trainY.shape, testY.shape

#print(trainX [10])
def str_column_to_float(trainX, column):
	for row in dataset:
		row[column] = float(row[column].strip())
def str_column_to_float(testX, column):
	for row in dataset:
		row[column] = float(row[column].strip())
def str_column_to_float(trainY, column):
	for row in dataset:
		row[column] = float(row[column].strip())
def str_column_to_float(testY, column):
	for row in dataset:
		row[column] = float(row[column].strip())
        
trainY = np.genfromtxt(local_dir + 'mnist_y_train', max_rows=100) #very small sample used due to limitation of machine's RAM
trainX = np.genfromtxt(local_dir + 'mnist_x_train', max_rows=100) 


from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression(fit_intercept=True) #, C = 1e15)

logreg = logisticRegr.fit(trainX, trainY)

#Predict labels for new data (new images). Uses the information the model learned during the model training process
coefficents= logreg.coef_
print (logreg.intercept_), print(coefficents)
print(logreg.score(trainX, trainY)) # Returns the mean accuracy on the given test data and labels.

#logisticRegr.predict(testX[0].reshape(1,-1))
#logisticRegr.predict(testX[0:10]) #Predict for Multiple Observations (images) at Once
predictions = logisticRegr.predict(testX) #Make predictions on entire test data
print(predictions)

#b)
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def prediction_function_vec(X, coefficents):
    p1 = sigmoid((X@coefficents)) #predicted prob of label 1
    return p1

def prediction_function_vec(X, theta, threshold=0.5):
    return predict_prob(X,theta)>=threshold

# Making prediction with coefficients
def predict(row, coefficients):
	yhat = weights[0]
	for i in range(len(row)-1):
		yhat += weights[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))
  
  
#c)
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

#Finally, I’m ready to build the model function. I’ll add in the option to calculate the model with an intercept, since it’s a good option to have.
def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
        # Print log-likelihood every so often
        if step % 10000 == 0:
            print (log_likelihood(features, target, weights))
        
    return weights

#run the model
logistic_regression(features, target, num_steps, learning_rate, add_intercept = False)
weights = logistic_regression(trainX, trainY, num_steps = 1000, learning_rate = 5e-5, add_intercept=True)
print (weights)


#d)
# classification error
from sklearn.metrics import accuracy_score
my_accuracy = accuracy_score(trainX, trainY, normalize=False) / float(trainX.size) #create vectors for actual labels and predicted labels...

# To get the accuracy, I just need to use the final weights to get the logits for the dataset (final_scores).
#Then I can use sigmoid to get the final predictions and round them to the nearest integer (0 or 1) to get the predicted class.
data_with_intercept = np.hstack((np.ones((trainX.shape[0], 1)),  trainX))
final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))

print ('Accuracy from scratch: {0}'.format((preds == trainY).sum().astype(float) / len(preds)))
print ('Accuracy from sk-learn: {0}'.format(logreg.score(trainX, trainY)))

