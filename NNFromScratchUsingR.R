# Image classification is one important field in Computer Vision, 
# not only because so many applications are associated with it, but also a lot of Computer Vision problems can be effectively reduced to image classification. The state of art tool in image classification is Convolutional Neural Network (CNN). In this article, I am going to write a simple Neural Network with 2 layers (fully connected). First, I will train it to classify a set of 4-class 2D data and visualize the decision bounday. Second, I am going to train my NN with the famous MNIST data (https://www.kaggle.com/c/digit-recognizer) and see its performance. The first part is inspired by CS 231n course offered by Stanford: http://cs231n.github.io/, which is taught in Python.
# Data set generation
# First, let’s create a spiral dataset with 10 classes and 1000 examples each.
# setting a working dir
setwd("~/Dropbox/current/NNetFromScratchUsingR/")
# loading libraries
library(ggplot2)
library(caret) 

source("NNet.R")
source("NNetPred.R")

N <- 1000 # number of points per class
D <- 20 # dimensionality
K <- 10 # number of classes
X <- data.frame() # data matrix (each row = single example)
y <- data.frame() # class labels 

set.seed(308) 
for (j in (1:K)) {
  r <- seq(0.05, 1, length.out = N) # radius  
  t <- seq((j - 1) * 4.7, j * 4.7, length.out = N) + rnorm(N, sd = 0.3) # theta  
  Xtemp <- data.frame(x = r * sin(t), y = r * cos(t))   
  ytemp <- data.frame(matrix(j, N, 1))  
  X <- rbind(X, Xtemp)  
  y <- rbind(y, ytemp)
}

data <- cbind(X, y)
colnames(data) <- c(colnames(X), 'label')

# X, y are 800 by 2 and 800 by 1 data frames respectively, and they are created in a way 
# that a linear classifier cannot separate them. Since the data is 2D, 
# we can easily visualize it on a plot. They are roughly evenly spaced and indeed a line is not a good decision boundary.

x_min <- min(X[ , 1]) - 0.2; 
x_max <- max(X[ , 1]) + 0.2
y_min <- min(X[ , 2]) - 0.2; 
y_max <- max(X[ , 2]) + 0.2 

# lets visualize the data:
ggplot(data) + geom_point(aes(x = x, 
                              y = y, 
                              color = as.character(label)), 
                          size = 2) + 
  theme_bw(base_size = 15) +  
  xlim(x_min, x_max) + 
  ylim(y_min, y_max) +  
  ggtitle('Spiral Data Visulization') +  
  coord_fixed(ratio = 0.8) +  
  theme(axis.ticks = element_blank(), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),         
        axis.text = element_blank(), 
        axis.title = element_blank(), 
        legend.position = 'none')

# Neural network construction
# Now, let’s construct a NN with 2 layers. 
# But before that, we need to convert X into a matrix (for matrix operation later on). 
# For labels in y, a new matrix Y (800 by 4) is created such that 
# for each example (each row in Y), the entry with index==label is 1 (and 0 otherwise).

X <- as.matrix(X)
Y <- matrix(0, N * K, K) 
for (i in 1:(N * K))
  { Y[i, y[i, ]] <- 1 }

nnet.model <- NNet(X, Y, step_size = 0.4, reg = 0.0002, h = 50, niteration = 10000)

## [1] "iteration 0 : loss 1.38628868932674"
## [1] "iteration 100 : loss 0.967921639616882"
## ...
## [1] "iteration 6000 : loss 0.218468573259166"

predicted_class <- NNetPred(X, nnet.model)
print(paste('training accuracy:', mean(predicted_class == (y))))
## [1] "training accuracy: 0.96375"

# Decision boundary
# Next, let’s plot the decision boundary. We can also use the caret package and train different classifiers with the data and visualize the decision boundaries. It is very interesting to see how different algorithms make 
# plot the resulting classifier
hs <- 0.01
grid <- as.matrix(expand.grid(seq(x_min, x_max, by = hs), 
                              seq(y_min, y_max, by = hs)))

Z <- NNetPred(grid, nnet.model) 

ggplot() + geom_tile(aes(x = grid[ , 1], 
                         y = grid[ , 2], 
                         fill = as.character(Z)), 
                      alpha = 0.3, show.legend = F) +   
  geom_point(data = data, aes(x = x, y = y, 
                              color = as.character(label)), 
             size = 2) + 
  theme_bw(base_size = 15) +  
  ggtitle('Neural Network Decision Boundary') +  
  coord_fixed(ratio = 0.8) +   
  theme(axis.ticks=element_blank(), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),         
        axis.text = element_blank(), 
        axis.title = element_blank(), 
        legend.position = 'none')

# MNIST data and preprocessing
# The famous MNIST (“Modified National Institute of Standards and Technology”) dataset is a classic 
# within the Machine Learning community that has been extensively studied. 
# It is a collection of handwritten digits that are decomposed into a csv file, 
# with each row representing one example, and the column values are grey scale from 0-255 of each pixel. 

# First, let’s display an image.

displayDigit <- function(X) {   
  m <- matrix(unlist(X), nrow = 28, byrow = T)  
  m <- t(apply(m, 2, rev))  
  image(m, col = grey.colors(255))
} 

train <- read.csv("data/train.csv", header = TRUE, stringsAsFactors = F)

displayDigit(train[18, -1])

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

#Model training and CV accuracy
# Now we can train the model with the training set. 
# Note even after removing nzv columns, the data is still huge,
# so it may take a while for result to converge. 
# Here I am only training the model for 3500 interations. 
# You can vary the iterations, learning rate and regularization strength 
# and plot the learning curve for optimal fitting.

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

displayDigit(Xtest)
