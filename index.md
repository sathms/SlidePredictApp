---
title       : Predict App - Using Random forest classifier
subtitle    : 
author      : sath_ms
job         : July 27, 2020
framework   : io2012        # {io2012, html5slides, shower, dzslides, ...}
highlighter : highlight.js  # {highlight.js, prettify, highlight}
hitheme     : tomorrow      # 
widgets     : []            # {mathjax, quiz, bootstrap}
mode        : selfcontained # {standalone, draft}
knit        : slidify::knit2slides
---

## Welcome to Random forest Classifier Demo

1. Build a shiny app to demonstrate a Random forest Classifier
2. For this we will use inbuilt mtcars dataset
3. We want to predict if a car is Manual or Automatic
4. To build a model user select inputs  
    Input1: Pick a number for k-fold cross validation; 
    Input2: Select one or more predictors
5. The app splits the data 70:30 for training and validation
6. The training data is fed through a model built using Random Forest Classifier
7. The validation data is then run through the trained model
8. The model is then evaluated for its accuracy vs ground truth
9. User can try different predictors and see if results change

Checkout app at https://sath-ms.shinyapps.io/PredictApp/



---

## Data Preprocessing


```r
library(shiny)
library(ggplot2)
library(caret)
library(lattice)
library(randomForest)
library(e1071)

data(mtcars)
mdata <- mtcars
mdata$am <- factor(mdata$am, labels = c("Automatic", "Manual"))
set.seed(7826)
inTrain <- createDataPartition(mdata$am, p = 0.7, list = FALSE)
train <- mdata[inTrain, ]
valid <- mdata[-inTrain, ]
```

---

# Get User Inputs, build a classifier model with random forest classifier


```r
kfold <-  5     #  User can select: 2, 5 or 10
# User can select one or more predictors from the following list
predictors <- c("mpg","cyl","disp","hp","drat","wt","qsec","vs","gear","carb")
kfold
```

```
## [1] 5
```

```r
predictors
```

```
##  [1] "mpg"  "cyl"  "disp" "hp"   "drat" "wt"   "qsec" "vs"   "gear" "carb"
```


```r
control <- trainControl(method = "cv", number = kfold)
formulaText <-paste("am ~", paste(predictors, collapse = " + "), collapse = " ")
model <- train(as.formula(formulaText),
               data = train,
               method = "rf",
               trControl = control)
model
```

```
## Random Forest 
## 
## 24 samples
## 10 predictors
##  2 classes: 'Automatic', 'Manual' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 19, 19, 19, 20, 19 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa    
##    2    0.84      0.6904762
##    6    0.88      0.7662338
##   10    0.92      0.8321678
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 10.
```

---

# Run the model with validation dataset. Review the model output vs ground truth.


```r
predict_rf <- predict(model, valid)
conf_rf <- confusionMatrix(valid$am, predict_rf)
as.data.frame.table(conf_rf$table)
```

```
##   Prediction Reference Freq
## 1  Automatic Automatic    4
## 2     Manual Automatic    0
## 3  Automatic    Manual    1
## 4     Manual    Manual    3
```

```r
conf_rf$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##     0.87500000     0.75000000     0.47349033     0.99684028     0.50000000 
## AccuracyPValue  McnemarPValue 
##     0.03515625     1.00000000
```
