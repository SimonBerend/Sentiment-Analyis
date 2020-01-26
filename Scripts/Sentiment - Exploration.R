
# -------------------------------------------------------------------------
# GOAL :
# DESCRIPTION :
# DEVELOPER : BEREND
# Wed Jan 22 14:57:45 2020 ------------------------------
# -------------------------------------------------------------------------

# Pacman -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, e1071, caret, doParallel,
               plotly, magrittr, ggplot2, corrplot,
               pls, kknn, e1071)


# Get Data ----------------------------------------------------------------
galaxy <- read.csv("Data/galaxy_smallmatrix_labeled_8d.csv")
apple <- read.csv("Data/iphone_smallmatrix_labeled_8d.csv")


# Relevant columns only ---------------------------------------------------
# Note : disregard iOS data
galaxy %<>% select(starts_with("samsung"), contains("senti"))
apple %<>% dplyr::select(starts_with("iphone"))


# Distribution of sentiments -------------------------------------------------------------------------
plot_ly(galaxy, x= ~galaxy$galaxysentiment, type='histogram', name = "galaxy") %>% 
  add_trace(data = apple, x = apple$iphonesentiment, name = "apple")


# All set -------------------------------------------------------------------------

appleAll <- apple


# Correlation / Collinearity -------------------------------------------------------------------------
cor(apple, apple$iphonesentiment)
# cor(galaxy, galaxy$galaxysentiment)


cor_app <- cor(apple)
cor_gal <- cor(galaxy)

cor_app[cor_app > 0.8] %>% print()

corrplot(cor_app)
corrplot(cor_gal)

# iphonedisunc is highy correlated with both iphonedispos and iphonedisneg
# although value is not > .9  , you could take out this variable
# cor_app[c(4:6), c(4:6)]
# apple$iphonedisunc <- NULL


# Near Zero Variance ---------------------------------------------------------------------

nzvApple <- nearZeroVar(apple, saveMetrics = T)
nzvGalaxy <- nearZeroVar(galaxy, saveMetrics = T)
nzv <- nearZeroVar(apple, saveMetrics = F)

# iphonecamneg [, 3] comes out as a variable with near zero variance
apple <- apple[, -nzv]


# CARET::RFE -------------------------------------------------------------------------
# sentiment as ordered factor
apple$iphonesentiment  %<>% factor(ordered = T)

# Let's sample the data before using RFE
set.seed(123)
appSample <- apple[sample(1:nrow(apple), size = 1000, replace = FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable
start_time <- Sys.time()

rfeResults <- rfe(appSample[, !names(appSample) == "iphonesentiment"], 
                  appSample$iphonesentiment, 
                  sizes = (1:9), 
                  rfeControl=ctrl)

end_time <- Sys.time()
end_time - start_time

# Get / plot results
# rfeResults
# plot(rfeResults, type = c("g", "o"))

# create data set with rfe recommended features
appleRFE <- apple[, predictors(rfeResults)]
# add the dependent variable to iphoneRFE
appleRFE$iphonesentiment <- apple$iphonesentiment


# -------------------------------------------------------------------------
# create sample sets
allSample <- appleAll[sample(1:nrow(appleAll), size = 1000, replace = FALSE),]
rfeSample <- appleRFE[sample(1:nrow(appleRFE), size = 1000, replace = FALSE),]

# create data partition
inTrain <- createDataPartition(y = allSample$iphonesentiment, p = .75, list = FALSE)

# use data partition on both data sets
# All set
allTrain <- allSample[inTrain, ]
allTest <- allSample[-inTrain, ]
# rfe set
rfeTrain <- rfeSample[inTrain, ]
rfeTest <- rfeSample[-inTrain, ]

# modify resampling method
ctrl <- trainControl(method = "repeatedcv",
                     verboseIter = TRUE,
                     repeats = 2,
                     number = 4
                    )

# RFE MODEL :  RF--------------------------------------------------------------
start_time <- Sys.time()

model.rfe.rf.tune <- train(iphonesentiment ~ .,
                      data = rfeTrain,
                      method = "rf", 
                      # tuneLength = 2,
                      tuneGrid = expand.grid(mtry = c(2, 3, 4)),
                      # tuneGrid = expand.grid(n = c(1:10)),
                      trControl = ctrl,
                      preProc = c("center", "scale")
)

end_time <- Sys.time()
end_time - start_time


# RFE MODEL :  kknn --------------------------------------------------------------
start_time <- Sys.time()

model.rfe.kknn <- train(iphonesentiment ~ .,
                       data = rfeTrain,
                       method = "kknn", 
                       #tuneLength = 20,
                       tuneGrid = expand.grid(kmax = c(28, 35, 42), distance = c(1, 3, 5), kernel="optimal" ),
                       trControl = ctrl,
                       preProc = c("center", "scale")
)

end_time <- Sys.time()
end_time - start_time

# Check Model Metrics -----------------------------------------------------
# collect resamples
results <- resamples(list(C5 =  model.rfe.c5,
                          RandomForest = model.rfe.rf,
                          SVM = model.rfe.svm, 
                          KKNN = model.rfe.kknn))

# summarize the distributions
summary(results)

# boxplot of results
bwplot(results)

# Inspect Performance -----------------------------------------------------
plot(model.rfe.c5)
plot(model.rfe.rf)

plot(varImp(model.rfe.c5), main = "C5.0")
plot(varImp(model.rfe.rf), main = "rf")


# Run Predictions and check real metrics ---------------------------------------------------------
# run predictions
predict_c5 <- predict(model.rfe.c5, rfeTest)
predict_rf <- predict(model.rfe.rf, rfeTest)
predict_svm <- predict(model.rfe.svm, rfeTest)
predict_kknn <- predict(model.rfe.kknn, rfeTest)


# check performance
yes <- list(rf = postResample(predict_rf, rfeTest$iphonesentiment),
            c5 = postResample(predict_c5, rfeTest$iphonesentiment),
            svm = postResample(predict_svm, rfeTest$iphonesentiment),
            kknn = postResample(predict_kknn, rfeTest$iphonesentiment)
                  )

cm_list <- list(rf = confusionMatrix(predict_rf, rfeTest$iphonesentiment),
                c5 = confusionMatrix(predict_c5, rfeTest$iphonesentiment)
                )
                


# Check RF Model on "All" set ------------------------------------------
start_time <- Sys.time()

model.all.rf.tune <- train(iphonesentiment ~ .,
                      data = allTrain,
                      method = "rf", 
                      # tuneLength = 2,
                      tuneGrid = expand.grid(mtry = c(2, 3, 4)),
                      # tuneGrid = expand.grid(n = c(1:10)),
                      trControl = ctrl,
                      preProc = c("center", "scale")
)

end_time <- Sys.time()
end_time - start_time


# compare results of rf models ---------------------------------------------------------
# Check Model Metrics -----------------------------------------------------
# collect resamples
results.rf <- resamples(list(All =  model.all.rf,
                          rfe = model.rfe.rf,
                          all.tune = model.all.rf.tune,
                          rfe.tune = model.rfe.rf.tune))

results.rf$values


# boxplot of results
bwplot(results.rf, main = "Random Forest metrics")

# Inspect Performance -----------------------------------------------------
plot(model.all.rf.tune)
plot(model.all.rf)

plot(varImp(model.rfe.rf.tune), main = "tune")
plot(varImp(model.rfe.rf), main = "rf")


predict_rf_tune <- predict(model.rfe.rf.tune, rfeTest)

confusionMatrix(predict_rf, rfeTest$iphonesentiment)
confusionMatrix(predict_rf_tune, rfeTest$iphonesentiment)


# Go for the tuned rfe rf model and check see if you can balance out the data set


# Feature selection : recode() the dependent variable ---------------------
# create a new dataset that will be used for recoding sentiment
appleRC <- appleRFE
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
appleRC$iphonesentiment <- recode(appleRC$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# set as ordered factor
appleRC$iphonesentiment %<>% factor(ordered = TRUE)


saveRDS(appleRC, "Data/appleRC.rds")

# Distinct cases only, to balance out data set ------------------------
distinct()



