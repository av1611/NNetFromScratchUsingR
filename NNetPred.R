# Next, create a prediction function, which takes X (same col as training X but may have different rows) and layer parameters as input. The output is the column index of max score in each row. In this example, the output is simply the label of each class. Now we can print out the training accuracy.

NNetPred <- function(X, para = list()) {  
  W <- para[[1]]  
  b <- para[[2]]  
  W2 <- para[[3]]  
  b2 <- para[[4]]  
  N <- nrow(X)  
  hidden_layer <- pmax(0, X %*% W + matrix(rep(b,N), nrow = N, byrow = T))   
  hidden_layer <- matrix(hidden_layer, nrow = N)  
  scores <- hidden_layer %*% W2 + matrix(rep(b2,N), nrow = N, byrow = T)  
  predicted_class <- apply(scores, 1, which.max)   
  return(predicted_class)  
} 
