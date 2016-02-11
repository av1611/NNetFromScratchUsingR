setwd("~/Dropbox/current/NNetFromScratchUsingR/")
# loading libraries
library(ggplot2)
library(caret) 

source("NNet.R")
source("NNetPred.R")
source("DisplayDigit.R")

train <- read.csv("data/train.csv", header = TRUE, stringsAsFactors = F)

DisplayDigit(train[18, -1])

# Now, let’s preprocess the data by removing near zero variance columns and scaling by max(X). The data is also splitted into two for cross validation. Once again, we need to creat a Y matrix with dimension N by K. This time the non-zero index in each row is offset by 1: label 0 will have entry 1 at index 1, label 1 will have entry 1 at index 2, and so on. In the end, we need to convert it back. (Another way is put 0 at index 10 and no offset for the rest labels.)

nzv <- nearZeroVar(train)
nzv.nolabel <- nzv - 1 
inTrain <- createDataPartition(y = train$label, p = 0.7, list = F) 
training <- train[inTrain, ]
CV <- train[-inTrain, ] 
X <- as.matrix(training[ , -1]) # data matrix (each row = single example)
N <- nrow(X) # number of examples
y <- training[, 1] # class labels 
K <- length(unique(y)) # number of classes
X.proc <- X[, -nzv.nolabel] / max(X) # scale
D <- ncol(X.proc) # dimensionality X
cv <- as.matrix(CV[ , -1]) # data matrix (each row = single example)
Xcv <- as.matrix(CV[ , -1])
ycv <- CV[, 1] # class labels
Xcv.proc <- Xcv[ , -nzv.nolabel]/max(X) # scale CV data 
Y <- matrix(0, N, K) 

for (i in 1:N) 
{ Y[i, y[i] + 1] <- 1 }

nnet.mnist <- NNet(X.proc, Y, step_size = 0.3, reg = 0.0001, niteration = 5000)

## [1] "iteration 0 : loss 2.30265553844748"
### ...
## [1] "iteration 3500 : loss 0.250350279456443"

predicted_class <- NNetPred(X.proc, nnet.mnist)
print(paste('training set accuracy: ', mean(predicted_class == (y+1))))

## [1] "training set accuracy: 0.93089140563888"

predicted_class <- NNetPred(Xcv.proc, nnet.mnist)
print(paste('CV accuracy: ', mean(predicted_class == (ycv+1))))



# Prediction of a random image
# Randomly selecting an image and predicting the label.

Xtest <- Xcv[sample(1:nrow(Xcv), 1), ]
Xtest.proc <- as.matrix(Xtest[-nzv.nolabel], nrow = 1)
predicted_test <- NNetPred(t(Xtest.proc), nnet.mnist)
print(paste('The predicted digit is: ',predicted_test-1 ))

DisplayDigit(Xtest)
