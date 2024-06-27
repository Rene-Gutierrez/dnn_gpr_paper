# Deep Neural Network Analysis

# Libraries
library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)

# Data Cleaner
staDat <- TRUE
source("./data_cleaner.R")

# Data Transformed for DNN
traInpDat <- matrix(data = NA, nrow = traSiz, ncol = numRoi * 2)
for(i in 1:traSiz){
  traInpDat[i,] <- c(traDat[c(2, 4),i,])
}
traOutDat <- traDat[1,,]
tesInpDat <- matrix(data = NA, nrow = tesSiz, ncol = numRoi * 2)
for(i in 1:tesSiz){
  tesInpDat[i,] <- c(tesDat[c(2, 4),i,])
}
tesOutDat <- tesDat[1,,]

# Model Initialization
model <- keras_model_sequential()
model %>%
  layer_dense(units = 80, activation = 'relu', input_shape = c(2 * numRoi)) %>%
  layer_dense(units = 40, activation = 'relu') %>%
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dense(units = 40, activation = 'relu') %>%
  layer_dense(units = 80, activation = 'relu') %>%
  layer_dense(units = 2 * numRoi)


# Model Compilation
model %>% compile(loss      = 'mse',
                  optimizer = 'rmsprop', 
                  metrics   = 'mae') 

# Model Fit
mymodel <- model %>%          
  fit(x                = traInpDat,
      y                = traInpDat,
      epochs           = 1000,
      batch_size       = 50,
      validation_split = 0.2)

# Model Test
model %>% evaluate(x = tesInpDat,
                   y = tesInpDat)

samFit <- model %>% predict(traInpDat)
preFit <- model %>% predict(tesInpDat)

samErr <- colMeans((traInpDat - samFit)^2)
preErr <- colMeans((tesInpDat - preFit)^2)

print(paste0("Train Error: ", mean(samErr)))
print(paste0("Test Error: ", mean(preErr)))
