

---
title: "Peer-graded Assignment: Prediction Assignment Writeup"
author: "Shovit Bhari"
date: 2020-05-25
output: html_document
---



# Human Activity Recognition

## 1. Overview

This is the final report for Coursera's Practical Machine Learning course, which is a part of Data Science Specializaiton. The main goal of the project is to predict the manner in which 6 participants performed dumbell bicep curls using different machine learning algorithms. The results and accuracy of the training model is used to predict the behavior of 20 test cases. The results of the predictions are submitted in a quiz.  

## 2. Introduction

Large amount of data about personal activity can be collected relatively inexpensively using devices such as Jawbone Up, Nike FuleBand, and Fitbit. These smart devices have helped improve people's health by finding patterns in their behaviors. People often quntify how much of a particular activity they do, but rarely quantify their quality. 

As per [Human Activity Recognition Project,](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har) Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions and accelerometers are located in different parts. 

* Class A: Exactly according to the specification (Class A) 
* Class B: Throwing the elbows to the front
* Class C: Lifting the dumbell only halfway
* Class D: Lowering the dumbell only halfway
* Class E: Throwing the hips to the front

The location of accelerometers are depicted in the image below. 

![Image Credit:http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har](onbody.png){width=200px}

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. Researchers made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg). 


## 3. Environment Preparation

We first upload the R libraries that are necessary for the complete analysis.

```r
library(knitr)
library(caret)
library(ggplot2)
library(randomForest)
library(rpart)
library(rpart.plot)
library(rattle)
library(parallel)
library(doParallel)
library(corrplot)
```

## 4. Data Preparation
### 4.1. Download data

```r
if(!dir.exists("./data")){dir.create("./data")}
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
filepath1 <- "./data/pml_training.csv"
filepath2 <- "./data/pml_testing.csv"
download.file (url1, filepath1)
download.file (url2, filepath2)
```

### 4.2. Load Data

```r
training <- read.csv("./data/pml_training.csv", na.strings = c("NA", "#DIV/0!", ""))
testing <- read.csv("./data/pml_testing.csv", na.strings= c("NA", "#DIV/0!", ""))
dim(training); dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

### 4.3. Clean Data
Both datasets have 160 variables and have plenty of variables with NA values. We first remove those data that contains more than 95% of the observation ot be NA. We filter out those records. We remove NA values are removed and datasets with no NA's are created. In addition, we remove non-predictors which includes the index, subject anme, time and window variables.

```r
noNA <- colSums(is.na(training))/nrow(training) < 0.95
training_noNA <- training[, noNA]
testing_noNA <- testing[,noNA]
training_new <- training_noNA[,-c(1:7)]
testing_new <- testing_noNA[,-c(1:7)]
dim(training_new);dim(testing_new)
```

```
## [1] 19622    53
```

```
## [1] 20 53
```

### 4.4 Partition the training data into training and cross validation
The new training dataset is partitioned to 70% train dataset and 30% cross validation dataset. 

```r
inTrain  <- createDataPartition(training_new$classe, p=0.7, list=FALSE)
TrainSet <- training_new[inTrain, ]
CVSet  <- training_new[-inTrain, ]
dim(TrainSet)
```

```
## [1] 13737    53
```

### 4.5 A correlation among variables is analyzed modeling. 


```r
corrPlot <- cor(TrainSet[, -length(names(TrainSet))])
corrplot(corrPlot, method="color",type = "lower", 
         tl.cex = 0.9, tl.col = rgb(0, 0, 1))
```

![plot of chunk unnamed-chunk-1](figure/unnamed-chunk-1-1.png)
  

The highly correlated variables are shown in dark colors in the graph above.

## 5. Prediction Model Building

Three methods will be applied to the model to make prediction and confusion matrix is used to calculate the results and the accuracy.

5.1: Decision Tree  
5.2: Generalized Boosted Model  
5.3: Random Forest  

### 5.1 ML Algorithm:Decision Tree


```r
dtmodel <-train(classe~., data=TrainSet, method="rpart")
dtpred <- predict(dtmodel, CVSet)
decisionTreeCM <- confusionMatrix(CVSet$classe, dtpred)
decisionTreeCM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1514   34  122    0    4
##          B  454  416  269    0    0
##          C  468   48  510    0    0
##          D  418  185  361    0    0
##          E  153  150  276    0  503
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5001          
##                  95% CI : (0.4872, 0.5129)
##     No Information Rate : 0.511           
##     P-Value [Acc > NIR] : 0.9537          
##                                           
##                   Kappa : 0.3472          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5035  0.49940  0.33160       NA  0.99211
## Specificity            0.9444  0.85689  0.88130   0.8362  0.89234
## Pos Pred Value         0.9044  0.36523  0.49708       NA  0.46488
## Neg Pred Value         0.6455  0.91214  0.78843       NA  0.99917
## Prevalence             0.5110  0.14155  0.26134   0.0000  0.08615
## Detection Rate         0.2573  0.07069  0.08666   0.0000  0.08547
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
## Balanced Accuracy      0.7239  0.67814  0.60645       NA  0.94222
```

```r
rpart.plot(dtmodel$finalModel)
```

![plot of chunk dt](figure/dt-1.png)

```r
dtAccuracy <- confusionMatrix(CVSet$class, dtpred)$overall['Accuracy']
dtAccuracy
```

```
## Accuracy 
## 0.500085
```


GBM and RF methods takes a long time to run. So parallel processing is used in caret package to increase processing time. Combination of prallel and doParallel packages are used to improve performance time. 

```r
# 1: Configure Parallel Processing
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
cluster
```

```
## socket cluster with 7 nodes on host 'localhost'
```

```r
registerDoParallel(cluster)
```


```r
#2: Configure trainControl object
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)
```


### 5.2 ML Algorithm: Generalized Boosted Model

```r
#3: Develop Training Model
gbmmodel <-train(classe~., data=TrainSet, method="gbm", trControl=fitControl)
```


```r
gbmpred<- predict(gbmmodel, CVSet)
GBM_CM <- confusionMatrix(CVSet$classe, gbmpred)
gbmAccuracy<- confusionMatrix(CVSet$class, gbmpred)$overall['Accuracy']
GBM_CM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1648   16    7    1    2
##          B   34 1076   26    2    1
##          C    0   32  984    6    4
##          D    2    2   32  923    5
##          E    4   19   10   12 1037
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9631         
##                  95% CI : (0.958, 0.9678)
##     No Information Rate : 0.2868         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9533         
##                                          
##  Mcnemar's Test P-Value : 3.842e-08      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9763   0.9397   0.9292   0.9778   0.9886
## Specificity            0.9938   0.9867   0.9913   0.9917   0.9907
## Pos Pred Value         0.9845   0.9447   0.9591   0.9575   0.9584
## Neg Pred Value         0.9905   0.9855   0.9846   0.9957   0.9975
## Prevalence             0.2868   0.1946   0.1799   0.1604   0.1782
## Detection Rate         0.2800   0.1828   0.1672   0.1568   0.1762
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9851   0.9632   0.9602   0.9847   0.9896
```

```r
gbmAccuracy
```

```
##  Accuracy 
## 0.9631266
```

### 5.3 ML Algorithm:Random Forest

```r
#3: develop rf training model
rfmodel <- train(classe~., data=TrainSet, method="rf", trControl=fitControl)
```


```r
rfpred <- predict(rfmodel, CVSet)
rfCM<- confusionMatrix(CVSet$classe, rfpred)
rfCM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    3 1135    1    0    0
##          C    0    7 1019    0    0
##          D    0    0   17  947    0
##          E    0    0    0    4 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9946          
##                  95% CI : (0.9923, 0.9963)
##     No Information Rate : 0.285           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9931          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9939   0.9826   0.9958   1.0000
## Specificity            1.0000   0.9992   0.9986   0.9966   0.9992
## Pos Pred Value         1.0000   0.9965   0.9932   0.9824   0.9963
## Neg Pred Value         0.9993   0.9985   0.9963   0.9992   1.0000
## Prevalence             0.2850   0.1941   0.1762   0.1616   0.1832
## Detection Rate         0.2845   0.1929   0.1732   0.1609   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9991   0.9965   0.9906   0.9962   0.9996
```

```r
rfAccuracy<- confusionMatrix(CVSet$class, rfpred)$overall['Accuracy']
rfAccuracy
```

```
##  Accuracy 
## 0.9945624
```


```r
#4: de-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
```

## 6.Applying the Selected Model to the Test Data

The accuracy of the 3 regression modeling methods above are:

* Decision Tree : 0.500085
* GBM           : 0.9631266
* Random Forest : 0.9945624

Random Forest method has the highest accuracy of 0.9945624 and the error rate is low. Therefore, the Random Forest model will be applied to predict the testing dataset (Project Quiz results for 20 sets)


```r
predict(rfmodel, testing_new)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
