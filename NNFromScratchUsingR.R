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

x_min <- min(X[ , 1]) - 0.2; 
x_max <- max(X[ , 1]) + 0.2
y_min <- min(X[ , 2]) - 0.2; 
y_max <- max(X[ , 2]) + 0.2 

# let's take a look on the data:
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
# Next, let’s plot the decision boundary. 
# We can also use the caret package and train different classifiers with the data 
# and visualize the decision boundaries.
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

