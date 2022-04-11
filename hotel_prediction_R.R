library(dummies)
library(car)
library(olsrr)
library(caret)
library(lawstat)
library(lmtest)
library(corrplot)
library(tidyverse)
library(e1071)
library(fastDummies)
library(pROC)

# hotel booking is table name 
h <- hotel_bookings

# Filter the data by using country column
h <- h %>% filter(country == "PRT")

# order by date column
h <- h[order(h$reservation_status_date),]

# converting catagorical label values to numeric
h$is_canceled[h$is_canceled== 0] <- 'no'
h$is_canceled[h$is_canceled== 1] <- 'yes'

#To numeric
h$is_canceled <- as.factor(h$is_canceled)

# y is array of labels
y <- h$is_canceled

#remove response variable
h <- subset(h, select = -is_canceled)

#Pre-processing 
#there are 6 nearzero variance variables 
#observing them country and company has to eliminate
h <- subset(h, select = -country)
h <- subset(h, select = -company)

#Dummy varibales

#Finding null values
sapply(h, function(x) sum(is.na(x)))
#showing childrens having 4 NA values but other columns has NULL values 

#Replacing NULL to NA
h$agent[h$agent == "NULL"] <- NA

#now showing missing values
sapply(h, function(x) sum(is.na(x)))

#Agent has more than 10000 missing values 
#filling the missing values from the above column value
h$children <- na.locf(h$children)
h$agent<- na.locf(h$agent)

#looking for nulls
#sapply(h, function(x) sum(is.na(x)))


#Dummy variables

h <- dummy_cols(h, select_columns = 'meal', remove_selected_columns= TRUE)
h <- dummy_cols(h, select_columns = 'market_segment', remove_selected_columns= TRUE)
h <- dummy_cols(h, select_columns = 'distribution_channel', remove_selected_columns= TRUE)
h <- dummy_cols(h, select_columns = 'is_repeated_guest', remove_selected_columns= TRUE)
h <- dummy_cols(h, select_columns = 'reserved_room_type', remove_selected_columns= TRUE)
h <- dummy_cols(h, select_columns = 'deposit_type', remove_selected_columns= TRUE)
h <- dummy_cols(h, select_columns = 'customer_type', remove_selected_columns= TRUE)
h <- dummy_cols(h, select_columns = 'reservation_status', remove_selected_columns= TRUE)
h <- dummy_cols(h, select_columns = 'assigned_room_type', remove_selected_columns= TRUE)

#convert to numeric columns
encode <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

#coverting month column into ordinal encoding
h$arrival_date_month <- encode(h$arrival_date_month, order = c('January', 'February', 'March','April','May','June','July','August','September','October','November','December'))

#Converting hotel column to numeric form catagorical
h$hotel <- as.numeric(c("City Hotel" = "0", "Resort Hotel" = "1")[h$hotel])

#converting agent column to numeric
h$agent <- sapply(h$agent, as.numeric)
#Removing near-zero variance variable
h <- subset(h, select = -reservation_status_date)

#Correlation 

correlations <- cor(h)
#correlations
#dim(correlations)
#correlations[1:4, 1:4]
library(caret)

## To visually examine the correlation structure of the data, the corrplot package

library(corrplot)
#par(mfrow = c(1, 1)) 


#corrplot(correlations, order = "hclust")

highCorr <- findCorrelation(correlations, cutoff = .80)
#warnings()

colnames(h[,c(50, 57 ,34 ,33, 55 ,39 ,66  ,5 ,47 ,71)])
h_filtered <- h[, -highCorr]
length(h_filtered)

#column numbers of them 12 and 15
numeric_cols <- h_filtered[,c(2,6,7,8,9,10,11,12,13,15,16,17,18)]


# Histograms of numeric variables
par(mfrow = c(3, 3)) 
for (i in 1:ncol(numeric_cols)){
hist(numeric_cols[ ,i], xlab = names(numeric_cols[i]), main = paste(names(numeric_cols[i]), "Histogram"), breaks = 50)}


#boxplots of numeric variables

for (i in 1:ncol(numeric_cols)){
boxplot(numeric_cols[ ,i], ylab = names(numeric_cols[i]),horizontal=T,
         main = paste(names(numeric_cols[i]), "Boxplot"), col="steelblue")}

#As we observe skewness in the numeric variable
# Absorving after transformation the skewness is decreased but outliers still present
#---------------------------------------------------------------------------------------------------------------------------
# Trying boxcox for decreasing skewness
xConttrans<-preProcess(numeric_cols,method=c("center","scale","BoxCox","spatialSign")) 
xCont<-predict(xConttrans,numeric_cols)

skewValues_Boxcox <- apply(xCont, 2, skewness)

print(skewValues_Boxcox)
#-----------------------------------------------------------------------------------------------------------------------------------
xcata <- h_filtered[,-c(2,6,7,8,9,10,11,12,13,15,16,17,18)]
degenerate_cata<-nearZeroVar(xcata) 
degenerate_cata
colnames(xcata[,c(7,  9, 10 ,11, 12, 18, 19 ,21, 24, 25, 27, 28 ,29 ,30 ,31, 33 ,34, 35 ,38, 40, 41, 44, 45, 46, 47, 48)])
length(degenerate_cata)
length(which(xcata[,13]==0))
h_filtered_cata <- xcata[, -degenerate_cata]
length(h_filtered_cata)


#----------------------------------------
h_filtered_cata<- select(h_filtered_cata, -19)

x_new<-cbind(xCont,h_filtered_cata) 
#x_new = select(x_new, -31)

trainingRows <- createDataPartition(y, p = .62, list= FALSE)
#head(trainingRows)
nrow(trainingRows)

#y_ <- y$is_canceled

trainpredictors <- x_new[trainingRows, ]
trainclasses <- y[trainingRows]
nrow(trainpredictors)

# Do the same for the test set using negative integers.
testpredictors <- x_new[-trainingRows, ]
testclasses <- y[-trainingRows]
nrow(testpredictors)
nrow(trainpredictors)

#-------------------------Logistic Regression-----------------


ctrl <- trainControl(method= 'LGOCV', summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)

set.seed(1)

lrFull <- train(trainpredictors,
                y = trainclasses,
                method = "glm",
                preProcess = c("center","scale"),
                metric = "kappa",
                trControl = ctrl)
warnings()
plot(lrFull)
lrFull
lrtrain <-predict(lrFull,trainpredictors)
postResample(pred=lrtrain,obs=trainclasses)
lrpredict<-predict(lrFull,testpredictors)
confusionMatrix(data=lrpredict,reference=testclasses)


varImp(lrFull)


#-----------------------------------------------------------------------------
#Linear Discriminant models
set.seed(47)
ctrl <- trainControl(method = 'LGOCV',
                     summaryFunction = defaultSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)


LDAFull <- train(x= trainpredictors,
                 y = trainclasses,
                 preProc = c('center','scale'),
                 method = "lda",
                 metric = "Roc",
                 trControl = ctrl)

LDAFull
ldatrain <-predict(LDAFull,trainpredictors)
postResample(pred=ldatrain,obs=trainclasses)
ldapredict<-predict(LDAFull,testpredictors)
confusionMatrix(data=ldapredict,reference=testclasses)
varImp(LDAFull)


#----------------------------------------------------------------------------------

### Partial Least Squares Discriminant Analysis ##

set.seed(47)
ctrl <- trainControl(method = 'LGOCV',summaryFunction = defaultSummary,
                     classProbs = TRUE)


plsFit2 <- train(x = trainpredictors,
                 y = trainclasses,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:5),
                 preProc = c("center","scale"),
                 metric = "kappa",
                 trControl = ctrl)
plsFit2
plot(plsFit2)



plstrain <-predict(plsFit2,trainpredictors)
postResample(pred=plstrain,obs=trainclasses)
plspredict<-predict(plsFit2,testpredictors)
confusionMatrix(data=plspredict,reference=testclasses)

varImp(plsFit2)
#------------------------------------------------------------------------------
## Penalized Models ###

set.seed(47)
ctrl <- trainControl(method = 'LGOCV',
                     summaryFunction = defaultSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)


glmnGrid <- expand.grid(.alpha = c(0, .1, .3, .5, .7, .9, 1),
                        .lambda = seq(.01, .2, length = 10))


glmnTuned <- train(x=trainpredictors,
                   y = trainclasses,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "kappa",
                   trControl = ctrl)

glmnTuned

plot(glmnTuned, plotType = "level")

glmtrain <-predict(glmnTuned,trainpredictors)
postResample(pred=glmtrain,obs=trainclasses)
glmpredict<-predict(glmnTuned,testpredictors)
confusionMatrix(data=glmpredict,reference=testclasses)

varImp(glmnTuned)

#---------------------Nearest shirken 
ctrl <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE)

## nscGrid <- data.frame(.threshold = 0:4)
nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))
set.seed(47)
nscTuned <- train(x = trainpredictors, 
                  y = trainclasses,
                  method = "pam",
                  preProc = c("center", "scale"),
                  tuneGrid = nscGrid,
                  metric = "ROC",
                  trControl = ctrl)

nscTuned
plot(nscTuned)
nsctrain <-predict(nscTuned,trainpredictors)
postResample(pred=nsctrain,obs=trainclasses)
nscpredict<-predict(nscTuned,testpredictors)
confusionMatrix(data=nscpredict,reference=testclasses)

#####################Non-linear models######################################

#-----------------------------KNN--------------------------------------------


ctrl <- trainControl(method= 'LGOCV', summaryFunction = defaultSummary,
                     classProbs = TRUE, allowParallel = TRUE)
set.seed(47)
knnFit <- train(x = trainpredictors, 
                y = trainclasses,
                method = "knn",
                metric = "kappa",
                preProc = c("center", "scale"),
                ##tuneGrid = data.frame(.k = c(83*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.k = 1:15),
                trControl = ctrl)

knnFit
plot(knnFit)
knntrain <-predict(knnFit,trainpredictors)
postResample(pred=knntrain,obs=trainclasses)
knnpredict<-predict(knnFit,testpredictors)
confusionMatrix(data=knnpredict,reference=testclasses)
confusionMatrix(data=knntrain,reference=trainclasses)

h$is_canceled[h$is_canceled== 0] <- 'no'
h$is_canceled[h$is_canceled== 1] <- 'yes'

varImp(knnFit)


No_of_neighbor <- knnFit$results[1]
Kappa_values <- knnFit$results[3]

plot(No_of_neighbor$k,Kappa_values$Kappa, type="l", col="blue", lwd=1, pch=15, xlab="No of neighbor", ylab="kappa values", main="kappa vs K", points='circle')
########################################################################################################################

#-------------------------------SVM------------------------------------------------

set.seed(47)
library(kernlab)
library(caret)

ctrl <- trainControl(method= 'LGOCV', summaryFunction = defaultSummary,
                     classProbs = TRUE, allowParallel = TRUE)

sigmaRangeReduced <- sigest(as.matrix(trainpredictors))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 6)))

svmRModel <- train(x = trainpredictors, 
                   y = trainclasses,
                   method = "svmRadial",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   #tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   #minstep=
                   trControl = ctrl)
svmRModel
plot(svmRModel)


svmtrain <-predict(svmRModel,trainpredictors)
postResample(pred=svmtrain,obs=trainclasses)
svmpredict<-predict(svmRModel,testpredictors)
confusionMatrix(data=svmpredict,reference=testclasses)


################################################################################


#------------------------------MDA-------------------------------------------------


ctrl <- trainControl(method= 'LGOCV',summaryFunction =defaultSummary,
                     classProbs = TRUE, allowParallel = TRUE)

set.seed(47)


mdaFit <- train(x = trainpredictors, 
                y = trainclasses,
                method = "mda",
                metric = "kappa",
                preProcess = c('scale','center'),
                tuneGrid = expand.grid(.subclasses = 1:15),
                trControl = ctrl)
mdaFit
plot(mdaFit)

mdatrain <-predict(mdaFit,trainpredictors)
postResample(pred=mdatrain,obs=trainclasses)
mdapredict<-predict(mdaFit,testpredictors)
confusionMatrix(data= mdapredict,reference=testclasses)
confusionMatrix(data= mdatrain,reference=trainclasses)

varImp(mdaFit)
mdaFit$results

No_of_subclasses <- mdaFit$results[1]
Kappa_values_mda <- mdaFit$results[3]

plot(No_of_subclasses$subclasses,Kappa_values_mda$Kappa, type="l", col="blue", lwd=1, pch=15, xlab="No of subclasses", ylab="kappa values", main="kappa vs subclasses")


#plot(No_of_neighbor$k,Kappa_values$Kappa)

#-------------------------------------------------------------------------------

#--------------------------RDA--------------------------------------------------


set.seed(47)
ctrl <- trainControl(method= 'LGOCV', summaryFunction = defaultSummary,
                     classProbs = TRUE,allowParallel = TRUE)

rdaFit <- train(x = trainpredictors, 
                y = trainclasses,
                method = "rda",
                metric = "kappa",
                tuneGrid = expand.grid(.lambda = seq(0.01,1,0.1), .gamma = seq(0.1,1,2)), 
                trControl = ctrl)
rdaFit
plot(rdaFit)

rdatrain <-predict(rdaFit,trainpredictors)
postResample(pred=rdatrain,obs=trainclasses)
rdapredict<-predict(rdaFit,testpredictors)
confusionMatrix(data= rdapredict, reference=testclasses)
confusionMatrix(data= rdatrain, reference=trainclasses)

varImp(rdaFit)


rdaFit$results

lambda_values <- rdaFit$results[1]
Kappa_values_rda <- rdaFit$results[4]

plot(lambda_values$lambda,Kappa_values_rda$Kappa, type="l", col="blue", xlab="lambda values", ylab="kappa values", main="kappa vs lambda")
#lambda_values

#--------------------------------------------------------------------------------
#--------------------------FDA -------------------------------------------------


marsGrid <- expand.grid(.degree = 1:3, .nprune = 20:30)
ctrl <- trainControl(method= 'LGOCV', summaryFunction = defaultSummary,
                     classProbs = TRUE, allowParallel = TRUE)

fdaTuned <- train(x = trainpredictors, 
                  y = trainclasses,
                  method = "fda",
                  preProcess = c("center", "scale"),
                  # Explicitly declare the candidate models to test
                  tuneGrid = marsGrid,
                  trControl = ctrl)
#trControl = trainControl(method = 'CV',number = 5))

fdaTuned
plot(fdaTuned)
fdaPred <- predict(fdaTuned, newdata = testpredictors)
confusionMatrix(data = fdaPred,reference = testclasses)
fdatrain <-predict(fdaTuned,trainpredictors)
postResample(pred=fdatrain,obs=trainclasses)
confusionMatrix(data = fdatrain,reference = trainclasses)
varImp(fdaTuned)

write.csv(fdaTuned$results,"C:\\Users\\surya\\Downloads\\fdadata.csv", row.names = FALSE)
################################################################################-

#-------------------------------NNET----------------------------------------------

library(nnet)
nnetGrid <- expand.grid(.size = 21:23, .decay = c( .01,0.1,1))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (33 + 1) + (maxSize+1)*2) ## 4 is the number of predictors
set.seed(47)
ctrl <- trainControl(method= 'LGOCV',summaryFunction = defaultSummary,
                     classProbs = TRUE,allowParallel = TRUE)

nnetFit <- train(x = trainpredictors, 
                 y = trainclasses,
                 method = "nnet",
                 metric = "kappa",
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 20,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit
plot(nnetFit)
plot(nnetFit$results$size, nnetFit$results$Kappa,type="l",group_by='size')
nnettrain <-predict(nnetFit,trainpredictors)
postResample(pred=nnettrain,obs=trainclasses)
nnpredict<-predict(nnetFit,testpredictors)
confusionMatrix(data=nnpredict,reference=testclasses)
confusionMatrix(data=nnettrain,reference=trainclasses)
varImp(nnetFit)



write.csv(dfg,"C:\\Users\\surya\\Downloads\\nnetdata.csv", row.names = FALSE)
#-------------------------------------------------------------------------------

#-----------------------------naive bayes---------------------------------------


install.packages("klaR")
library(klaR)
set.seed(47)
ctrl <- trainControl( summaryFunction = defaultSummary,
                      classProbs = TRUE,allowParallel = TRUE)



x_newnb <- x_new
colnames(x_newnb) <- c('a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ab','ac','ad','ae','af','ag','ah','ai')
trainingRows <- createDataPartition(y, p = .62, list= FALSE)
#head(trainingRows)
nrow(trainingRows)
trainpredictors <- x_newnb[trainingRows, ]
trainclasses <- y[trainingRows]
nrow(trainpredictors)

# Do the same for the test set using negative integers.
testpredictors <- x_newnb[-trainingRows, ]
testclasses <- y[-trainingRows]
nrow(testpredictors)
length(testclasses)



nbFit <- train( x = trainpredictors, 
                y = trainclasses,
                method = "nb",
                metric = "kappa",
                tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
                trControl = ctrl)
nbFit
nbtrain <-predict(nbFit,trainpredictors)
postResample(pred=nbtrain,obs=trainclasses)
nbpredict<-predict(nbFit,testpredictors)
confusionMatrix(data=nbpredict,reference=testclasses)
confusionMatrix(data=nbtrain,reference=trainclasses)


varImp(nbFit)