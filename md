Practical Machine Learning Project
---
title: "Quantified Self Movement Report"
author: "By Jaja Yogo"
date: "Thursday, August 20, 2015"
output: word_document
---

## Executive Summary
Utilizing devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These form of devices are part of the quantified self-movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbbell of six (6) participants as they perform barbell lifts correctly and incorrectly 5 different ways. 

**GitHub Repo: https://github.com/Rao48/Practical-Machine-Learning-Project** 


##Intoduction 
 Six young participants aged between 20-28 years old were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different ways: * Class A - exactly according to the specification * Class B - throwing the elbows to the front * Class C - lifting the dumbbell only halfway * Class D - lowering the dumbbell only halfway * Class E - throwing the hips to the front Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied with the manner they were supposed to simulate. The Research scientists made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg). The objective of this report is to predict the manner in which subjects did the exercise. This is the “classe” variable in the training set. The model will use the other variables to predict with. This report describes; (i) how the model is built, (ii) use of cross validation and (iii) an estimate of expected out of sample error. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Getting and Cleaning Data
```{r}library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(rattle)
```

### Load Data

```{r}url_raw_training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"file_dest_training <- "pml-training.csv" 
#download.file(url=url_raw_training, destfile=file_dest_training, method="curl") url_raw_testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" file_dest_testing <- "pml-testing.csv" 
#download.file(url=url_raw_testing, destfile=file_dest_testing, method="curl")
```

####Import the data treating empty values as NA.
We will attempt to get reject observation with missing values(NA) in this step

```{r}mytraining <- read.csv(file_dest_training, na.strings=
c("NA",""), header=TRUE)
colnames_train <- colnames(mytraining)
mytesting <- read.csv(file_dest_testing, na.strings=
c("NA",""), header=TRUE)
colnames_test <- colnames(mytesting)
```

###Read and Clean the Data
After downloading the data from the data source, we can read the two csv files into two data frames.

```{r}dim(mytraining)
dim(mytesting)
sum(complete.cases(mytraining))
colnames(mytraining)
colnames(mytesting)```

####Remove columns with NA missing values
```{r}mytraining <- mytraining[, colSums(is.na(mytraining)) == 0] 
mytesting <- mytesting[, colSums(is.na(mytesting)) == 0]```

##### Verify that the column names (excluding classe and problem_id) are identical in the training and test set.

```{r}classe <- mytraining$classe
trainRemove <- grepl("^X|timestamp|window", names(mytraining))
mytraining <- mytraining[, !trainRemove]
trainCleaned <- mytraining[, sapply(mytraining, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(mytesting))
mytesting <- mytesting[, !testRemove]
testCleaned <- mytesting[, sapply(mytesting, is.numeric)]```

### Data Set Validation
The cleaned training data set contains 19622 observations and 53 variables. The testing data set contains 20 observations and 53 variables. The “classe” variable is still in the cleaned training set. Use validation data set for further evaluation. Search for covariates with almost without variability.


```{r}set.seed(11355)
idsless <- createDataPartition(y=mytraining$classe, p=0.25, list=FALSE)
myless1 <- mytraining[idsless,]
myremainder <- mytraining[-idsless,]
set.seed(11355)
idsless <- createDataPartition(y=myremainder$classe, p=0.33, list=FALSE)
myless2 <- myremainder[idsless,]
myremainder <- myremainder[-idsless,]
set.seed(11355)
idsless <- createDataPartition(y=myremainder$classe, p=0.5, list=FALSE)
myless3 <- myremainder[idsless,]
myless4 <- myremainder[-idsless,]

```


## Train Model and Algorith
We will trim the data for further analysis.
Divide each of these 4 sets into training (60%) and test (40%) sets.

```{r}set.seed(11355)# For reproducibile purpose
inTrain <- createDataPartition(y=myless1$classe, p=0.6, list=FALSE)
mylesstraining1 <- myless1[inTrain,]
mylesstesting1 <- myless1[-inTrain,]
set.seed(11355)
inTrain <- createDataPartition(y=myless2$classe, p=0.6, list=FALSE)
mylesstraining2 <- myless2[inTrain,]
mylesstesting2 <- myless2[-inTrain,]
set.seed(11355)
inTrain <- createDataPartition(y=myless3$classe, p=0.6, list=FALSE)
mylesstraining3 <- myless3[inTrain,]
mylesstesting3 <- myless3[-inTrain,]
set.seed(11355)
inTrain <- createDataPartition(y=myless4$classe, p=0.6, list=FALSE)
mylesstraining4 <- myless4[inTrain,]
mylesstesting4 <- myless4[-inTrain,]
```

## Evaluation of Data
We will use classification tree to evaluate data, on *train on training* set 1 of 4 with no extra features.

```{r}set.seed(11355)
modFit <- train(mylesstraining1$classe ~ ., data = mylesstraining1, method="rpart")
print(modFit, digits=3)
print(modFit$finalModel, digits=3)
# See fancyrpartplot in appendix section.```

### Data Analysis and Prediction for Test Data
Train on training set 1 of 4 with no additional items using "Random Forest".
Run against testing set 1 of 4 with no extra features.

```{r}predictions <- predict(modFit, newdata=mylesstesting1)
print(confusionMatrix(predictions, mylesstesting1$classe), 
digits=4)```

```{r}predictions <- predict(modFit, newdata=mylesstesting1)
print(confusionMatrix(predictions, mylesstesting1$classe), 
digits=4)```

#####Run against 20 testing set provided by class instruction.

```{r}print(predict(modFit, newdata=mytesting))```

###Estimation of model perfomance on validation data set
Then, we estimate the performance of the model on the validation data set. 
Train on training set 4 of 4 with only cross validation.

```{r}set.seed(11355)
modFit <- train(mylesstraining4$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training4)
print(modFit, digits=3)```

##### Run against 20 testing set provided in class.
```{r}print(predict(modFit, newdata=mytesting))```



## Conclusion
The out of sample error (error rate you get on new data set) in this analysis, it’s the error rate resulting after running the predict () function on the 4 testing sets. The testing set is approximately equal in size, thus in this analysis the random forest method was utilized in both preprocessing and cross validation against test sets 1-4 (set1:1 - .9714 = 0.0286, set21 - .9634 = 0.0366: set3:1 - .9655 = 0.0345, set4:1 - .9563 = 0.0437) yielding a predicted out of sample rate of 0.03585 when averaged. The analysis found three distinct predictions by appling the 4 models against the 20 item training set: A) Accuracy Rate 0.0286 Predictions: B A A A A E D B A A B C B A E E A B B B B) Accuracy Rates 0.0366 and 0.0345 Predictions: B A B A A E D B A A B C B A E E A B B B C) Accuracy Rate 0.0437 Predictions: B A B A A E D D A A B C B A E E A B B B


##Appendix: #

####I. Rattle FancyRpartPlot

```{r}fancyRpartPlot(modFit$finalModel)```

####II.  Decision Tree Visualization 
```{r}treeModel <- rpart(classe ~ ., data=mylesstraining1, method="class")
prp(treeModel) 
```

##Reference
1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
