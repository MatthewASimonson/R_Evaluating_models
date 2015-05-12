# Modeling methods

###################################
# Evaluating classification models:
###################################

# Building and applying a logistic regression spam model

setwd("~/Desktop/Data_Science/zmPDSwR-master/Spambase")


spamD <- read.table('spamD.tsv',header=TRUE,sep='\t')
spamTrain <- subset(spamD,spamD$rgroup=10)
spamTest <- subset(spamD,spamD$rgroup<10)
spamVars <- setdiff(colnames(spamD),list('rgroup','spam')) # column names excluding 'rgroup' and 'spam' (union, intersection, difference operation)
spamFormula <- as.formula(paste('spam=="spam"',paste(spamVars,collapse=' + '),sep=' ~ '))
spamModel <- glm(spamFormula,family=binomial(link='logit'),data=spamTrain)
spamTrain$pred <- predict(spamModel, newdata=spamTrain, type='response')
spamTest$pred <- predict(spamModel,newdata=spamTest, type='response')

# generate confusion matrix:
cM <- table(truth=spamTest$spam,prediction=spamTest$pred0.5)
 print(cM)
#prediction
#truth      FALSE TRUE
#non-spam   264   14
#spam        22  158

# ACCURACY = true / total; detection rate
# PRECISION = TP/(TP+FP) ; true positive per positive; detected correct rate
# RECALL = fraction of the things that are in the class are detected by the classifier; true positive / (false negative + true positive)
# F-score = P*A

# Plotting residuals

d <- data.frame(y=(1:10)^2,x=1:10)
model <- lm(y~x,data=d)
d$prediction <- predict(model,newdata=d)
library('ggplot2')
ggplot(data=d) + geom_point(aes(x=x,y=y)) +
  geom_line(aes(x=x,y=prediction),color='blue') +
  geom_segment(aes(x=x,y=prediction,yend=y,xend=x)) +
  scale_y_continuous('')

# Making a double density plot

ggplot(data=spamTest) +
  geom_density(aes(x=pred,color=spam,linetype=spam))

# Plotting the receiver operating characteristic curve
library('ROCR')
eval <- prediction(spamTest$pred,spamTest$spam)
plot(performance(eval,"tpr","fpr"))
print(attributes(performance(eval,'auc'))$y.values[[1]])

# Calculating log likelihood

sum(ifelse(spamTest$spam=='spam',
             log(spamTest$pred),
             log(1-spamTest$pred)))

sum(ifelse(spamTest$spam=='spam',
             log(spamTest$pred),
             log(1-spamTest$pred)))/dim(spamTest)[[1]]

# Title: Computing the null model’s log likelihood

 pNull <- sum(ifelse(spamTest$spam=='spam',1,0))/dim(spamTest)[[1]]
 sum(ifelse(spamTest$spam=='spam',1,0))*log(pNull) + sum(ifelse(spamTest$spam=='spam',0,1))*log(1-pNull)

# Title: Calculating entropy and conditional entropy

#   Define function that computes the entropy
#   from list of outcome counts
 entropy <- function(x) {
  xpos <- x[x0]
  scaled <- xpos/sum(xpos)
  sum(-scaled*log(scaled,2))
}

#   Calculate entropy of spam/non-spam
#   distribution
print(entropy(table(spamTest$spam)))
# [1] 0.9667165

#   Function to calculate conditional or
#   remaining entropy of spam distribution (rows)
#   given prediction (columns)
conditionalEntropy <- function(t) {
  (sum(t[,1])*entropy(t[,1]) + sum(t[,2])*entropy(t[,2]))/sum(t)
}

#   Calculate conditional or remaining entropy
#   of spam distribution given prediction
print(conditionalEntropy(cM))
# [1] 0.3971897

#  Clustering random data in the plane

set.seed(32297)
d <- data.frame(x=runif(100),y=runif(100))
clus <- kmeans(d,centers=5)
d$cluster <- clus$cluster

# Plotting our clusters

library('ggplot2');
library('grDevices')

h <- do.call(rbind,lapply(unique(clus$cluster),function(c) { f <- subset(d,cluster==c); f[chull(f),]}))
ggplot() +
  geom_text(data=d,aes(label=cluster,x=x,y=y,
                       color=cluster),size=3)  +
  geom_polygon(data=h,aes(x=x,y=y,group=cluster,fill=as.factor(cluster)),
               alpha=0.4,linetype=0) +
  theme(legend.position = "none")

#  Calculating the size of each cluster
table(d$cluster)

#  Calculating the typical distance between items in every pair of clusters

 library('reshape2')
 n <- dim(d)[[1]]
 pairs <- data.frame(
  ca = as.vector(outer(1:n,1:n,function(a,b) d[a,'cluster'])),
  cb = as.vector(outer(1:n,1:n,function(a,b) d[b,'cluster'])),
  dist = as.vector(outer(1:n,1:n,function(a,b)
    sqrt((d[a,'x']-d[b,'x'])^2 + (d[a,'y']-d[b,'y'])^2)))
)
 dcast(pairs,ca~cb,value.var='dist',mean)

#  Preparing the KDD data for analysis

#   Read the file of independent variables. All
#   data from
#   https://github.com/WinVector/zmPDSwR/tree/master/KDD2009.

setwd("~/Desktop/Data_Science/zmPDSwR-master/KDD2009")
d <- read.table('orange_small_train.data.gz',
                header=TRUE,
                sep='\t',
                na.strings=c('NA','')) 	#   Treat both NA and the empty string as missing
#   data.


churn <- read.table('orange_small_train_churn.labels.txt',header=FALSE,sep='\t') 	#   Read churn dependent variable.
d$churn <- churn$V1 	#    Add churn as a new column.

appetency <- read.table('orange_small_train_appetency.labels.txt',header=FALSE,sep='\t')

d$appetency <- appetency$V1 	#   Add appetency as a new column.
upselling <- read.table('orange_small_train_upselling.labels.txt',header=FALSE,sep='\t')

d$upselling <- upselling$V1 	#   Add upselling as a new column.
set.seed(729375)
#   By setting the seed to the pseudo-random
#   number generator, we make our work reproducible:
#   someone redoing it will see the exact same
#   results.

d$rgroup <- runif(dim(d)[[1]])
dTrainAll <- subset(d,rgroup<=0.9)
dTest <- subset(d,rgroup0.9) 	#   Split data into train and test subsets.
outcomes=c('churn','appetency','upselling')
vars <- setdiff(colnames(dTrainAll),
                c(outcomes,'rgroup'))
catVars <- vars[sapply(dTrainAll[,vars],class) %in%
                  c('factor','character')] 	#   Identify which features are categorical
#   variables.

numericVars <- vars[sapply(dTrainAll[,vars],class) %in%
                      c('numeric','integer')] 	#   Identify which features are numeric
#   variables.
rm(list=c('d','churn','appetency','upselling')) 	#   Remove unneeded objects from workspace.
outcome <- 'churn' 	#   Choose which outcome to model (churn).
pos <- '1' 	#   Choose which outcome is considered
#   positive.
useForCal <- rbinom(n=dim(dTrainAll)[[1]],size=1,prob=0.1)0 	#   Further split training data into training and
#   calibration.
dCal <- subset(dTrainAll,useForCal)
dTrain <- subset(dTrainAll,!useForCal)


# Plotting churn grouped by variable 218 levels

table218 <- table(
  Var218=dTrain[,'Var218'],   #   Tabulate levels of Var218.
  churn=dTrain[,outcome], 	#   Tabulate levels of churn outcome.
  useNA='ifany') 	#   Include NA values in tabulation.
print(table218)

#  Churn rates grouped by variable 218 codes

 print(table218[,2]/(table218[,1]+table218[,2]))


#  Function to build single-variable models for categorical variables

mkPredC <- function(outCol,varCol,appCol) {   # Note: 1
  pPos <- sum(outCol==pos)/length(outCol) 	# Note: 2
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[pos] 	# Note: 3
  vTab <- table(as.factor(outCol),varCol)
  pPosWv <- (vTab[pos,]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3) 	# Note: 4
  pred <- pPosWv[appCol] 	# Note: 5
  pred[is.na(appCol)] <- pPosWna 	# Note: 6
  pred[is.na(pred)] <- pPos 	# Note: 7
  pred 	# Note: 8
}

# Note 1:
#   Given a vector of training outcomes (outCol),
#   a categorical training variable (varCol), and a
#   prediction variable (appCol), use outCol and
#   varCol to build a single-variable model and then
#   apply the model to appCol to get new
#   predictions.
# (Predicted churn rate based on rate in observed categories of training set, same categories in test set, prediction is a function of levels of training variable and their respective rates of churn)

# Note 2:
#   Get stats on how often outcome is positive
#   during training.

# Note 3:
#   Get stats on how often outcome is positive for
#   NA values of variable during training.

# Note 4:
#   Get stats on how often outcome is positive,
#   conditioned on levels of training variable.

# Note 5:
#   Make predictions by looking up levels of
#   appCol.

# Note 6:
#   Add in predictions for NA levels of
#   appCol.

# Note 7:
#   Add in predictions for levels of appCol that
#   weren’t known during training.

# Note 8:
#   Return vector of predictions.

#  Applying single-categorical variable models to all of our datasets

for(v in catVars) {
  pi <- paste('pred',v,sep='')
  dTrain[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTrain[,v])
  dCal[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dCal[,v])
  dTest[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTest[,v])
}

#  Scoring categorical variables by AUC
library('ROCR')

 calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

 for(v in catVars) {
  pi <- paste('pred',v,sep='')
  aucTrain <- calcAUC(dTrain[,pi],dTrain[,outcome])
  if(aucTrain>=0.8) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f",
                  pi,aucTrain,aucCal))
  }
}

# Scoring numeric variables by AUC

 mkPredN <- function(outCol,varCol,appCol) {
  cuts <- unique(as.numeric(quantile(varCol,
                                     probs=seq(0, 1, 0.1),na.rm=T)))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}
 for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  dTrain[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTrain[,v])
  dTest[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTest[,v])
  dCal[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dCal[,v])
  aucTrain <- calcAUC(dTrain[,pi],dTrain[,outcome])
  if(aucTrain>=0.55) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f",
                  pi,aucTrain,aucCal))
  }
}

# Plotting variable performance

ggplot(data=dCal) +
  geom_density(aes(x=predVar126,color=as.factor(churn)))

# Running a repeated cross-validation experiment
var <- 'Var217'
aucs <- rep(0,100)
for(rep in 1:length(aucs)) {     #   For 100 iterations...
  useForCalRep <- rbinom(n=dim(dTrainAll)[[1]],size=1,prob=0.1)>0  	#   ...select a random subset of about 10% of the training data as hold-out set,...
  predRep <- mkPredC(dTrainAll[!useForCalRep,outcome],  	#   ...use the random 90% of training data to train model and evaluate that model on hold-out
                     dTrainAll[!useForCalRep,var],
                     dTrainAll[useForCalRep,var])
  aucs[rep] <- calcAUC(predRep,dTrainAll[useForCalRep,outcome])  	#   ...calculate resulting model’s AUC using hold-out set; store that value and repeat.
}
mean(aucs)
# [1] 0.5556656
sd(aucs)
# [1] 0.01569641

#  Empirically cross-validating performance

 fCross <- function() {
  useForCalRep <- rbinom(n=dim(dTrainAll)[[1]],size=1,prob=0.1)>0
  predRep <- mkPredC(dTrainAll[!useForCalRep,outcome],
                     dTrainAll[!useForCalRep,var],
                     dTrainAll[useForCalRep,var])
  calcAUC(predRep,dTrainAll[useForCalRep,outcome])
}
 aucs <- replicate(100,fCross())

# Basic variable selection

#    Each variable we use represents a chance of explaining
# more of the outcome variation (a chance of building a better
# model) but also represents a possible source of noise and
# overfitting. To control this effect, we often preselect
# which subset of variables we’ll use to fit. Variable
# selection can be an important defensive modeling step even
# for types of models that “don’t need it” (as seen with
# decision trees in section 6.3.2).  Listing 6.11 shows a
# hand-rolled variable selection loop where each variable is
# scored according to a deviance inspired score, where a
# variable is scored with a bonus proportional to the change
# in in scaled log likelihood of the training data.  We could
# also try an AIC (Akaike information criterion) by
# subtracting a penalty proportional to the complexity of the
# variable (which in this case is 2^entropy for categorical
# variables and a stand-in of 1 for numeric variables).  The
# score is a bit ad hoc, but tends to work well in selecting
# variables. Notice we’re using performance on the calibration
# set (not the training set) to pick variables. Note that we
# don’t use the test set for calibration; to do so lessens the
# reliability of the test set for model quality confirmation.

logLikelyhood <- function(outCol,predCol) {   #   Define a convenience function to compute log likelihood.
  sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
}

selVars <- c()
minStep <- 5
baseRateCheck <- logLikelyhood(dCal[,outcome],sum(dCal[,outcome]==pos)/length(dCal[,outcome]))

for(v in catVars) {
  #   Run through categorical variables and pick
  #   based on a deviance improvement (related to
  #   difference in log likelihoods; see chapter
  #   3).
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) - baseRateCheck))
  if(liCheck>minStep) {
    print(sprintf("%s, calibrationScore: %g",pi,liCheck))
    selVars <- c(selVars,pi)
  }
}

for(v in numericVars) {
  #   Run through numeric variables and pick
  #   based on a deviance improvement.
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) - baseRateCheck))
  if(liCheck>=minStep) {
    print(sprintf("%s, calibrationScore: %g",pi,liCheck))
    selVars <- c(selVars,pi)
  }
}


# Title: Building a bad decision tree

 library('rpart')
 fV <- paste(outcome,'>0 ~ ',paste(c(catVars,numericVars),collapse=' + '),sep='')
 tmodel <- rpart(fV,data=dTrain)
 print(calcAUC(predict(tmodel,newdata=dTrain),dTrain[,outcome]))
#[1] 0.9241265
 print(calcAUC(predict(tmodel,newdata=dTest),dTest[,outcome]))
#[1] 0.5266172
 print(calcAUC(predict(tmodel,newdata=dCal),dCal[,outcome]))
#[1] 0.5126917
