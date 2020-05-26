# PML-Project

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
