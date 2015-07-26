library(caret)
fileUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileUrl,destfile = "./train.csv",method="curl")
data<-read.csv("./train.csv",na.strings=c("NA","NaN", "  ","#DIV/0!"))
pattern<-c("kurtosis","skewness","max","min","amplitude","var","avg","stddev")
removeIndex<-grep(paste(pattern,collapse = "|"),colnames(data))
removeIndex<-c(c(1:7),removeIndex)
newdata<-data[,-removeIndex]

set.seed(123)

// to create 10 parts of data //
folds<-createFolds(newdata$class,k=10,list=TRUE,returnTrain = FALSE)

// use folds[1] to test if the data is linear or non-linear//
part1<-newdata[folds[[1]],]  
trainIndex<-createDataPartition(part1$classe,p=0.9,list=FALSE)
training<-part1[trainIndex,]
testing<-part1[-trainIndex,]

set.seed(617)
fitControl<-trainControl(method="repeatedcv",number=10,repeats=3)
set.seed(617)
ldafit<-train(classe~.,data=training,method="lda",trControl = fitControl)
confusionMatrix(training$classe,predict(ldafit,training))

set.seed(617)
qdafit<-train(classe~.,data=training,method="qda",trControl = fitControl)
confusionMatrix(training$classe,predict(qdafit,training))

set.seed(617)
knnfit<-train(classe~.,data=training,method="knn",trControl = fitControl,preProcess=c("center","scale"),tuneLength=20)
plot(knnfit)


treefit<-train(classe~.,data=training,method="rpart")
print(treefit$finalModel)
plot(treefit$finalModel, uniform=TRUE, main="Classification Tree")


gbmfit<-train(classe~.,data=training,method="gbm",trControl = fitControl)
confusionMatrix(training$classe,predict(gbmfit,training))

rffit<-train(classe~.,method="rf",data=training,prox=TRUE,trContril=fitControl)

bagfit<-train(classe~.,data=training,method="treebag",trControl = fitControl)

#compare perfermance
cvValues<-resamples(list(lda=ldafit,qda=qdafit,knn=knnfit,tree=treefit,bag=bagfit,gbm=gbmfit,rf=rffit))
summary(cvValues)


#testing
ldatest<-confusionMatrix(testing$classe,predict(ldafit,testing))
qdatest<-confusionMatrix(testing$classe,predict(qdafit,testing))
knntest<-confusionMatrix(testing$classe,predict(knnfit,testing))
treetest<-confusionMatrix(testing$classe,predict(treefit,testing))
bagtest<-confusionMatrix(testing$classe,predict(bagfit,testing))
rftest<-confusionMatrix(testing$classe,predict(rffit,testing))
gbmtest<-confusionMatrix(testing$classe,predict(gbmfit,testing))
data.frame(ldatest$overall,qdatest$overall,knntest$overall,treetest$overall,bagtest$overall,rftest$overall,gbmtest$overall)

predict(bagfit,testdata)
predict(gbmfit,testdata)


fileUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileUrl,destfile = "./test.csv",method="curl")
data<-read.csv("./test.csv",na.strings=c("NA","NaN", "  ","#DIV/0!"))
pattern<-c("kurtosis","skewness","max","min","amplitude","var","avg","stddev")
removeIndex<-grep(paste(pattern,collapse = "|"),colnames(data))
removeIndex<-c(c(1:7),removeIndex)
testdata<-data[,-removeIndex]