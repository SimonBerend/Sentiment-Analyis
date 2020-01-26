
# -------------------------------------------------------------------------
# GOAL : Continue Sentiment analysis 
# DESCRIPTION : See if feature selection and PCA in the prepreoces advance
# the model. Spoiler: PCA doesn't help. Accuracy does increase after feat engineering.
# NOTE:  The simplification of the dependent reduces the Kappa score.
# DEVELOPER : BEREND
# Sun Jan 26 15:45:42 2020 ------------------------------

# -------------------------------------------------------------------------

# Pacman -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, e1071, caret, doParallel,
               plotly, magrittr, ggplot2, corrplot,
               pls, kknn, e1071)


# Get Data ----------------------------------------------------------------
appleRC <- read_rds("Data/appleRC.rds")


# -------------------------------------------------------------------------
# set seed
set.seed(123)
# create sample sets
rcSample <- appleRC[sample(1:nrow(appleRC), size = 1000, replace = FALSE),]

# use data partition object 'inTrain' from Exploration script
rcTrain <- rcSample[inTrain, ]
rcTest <- rcSample[-inTrain, ]

# modify resampling method
#ctrl <- trainControl(method = "repeatedcv",
 #                    verboseIter = TRUE,
  #                   repeats = 2,
   #                  number = 4
    # )


# pre proces : PCA --------------------------------------------------------

# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(rcTrain[, !names(rcTrain) == "iphonesentiment"], 
                               method=c("center", "scale", "pca"),
                               thresh = 0.95)
# print(preprocessParams)


# Apply pca  --------------------------------------------------------------
# use predict to apply pca parameters, create train/test, exclude dependant
pcaTrain <- predict(preprocessParams, rcTrain[, !names(rcTrain) == "iphonesentiment"])
pcaTest <- predict(preprocessParams, rcTest[, !names(rcTrain) == "iphonesentiment"])

# add the dependent
pcaTrain$iphonesentiment <- rcTrain$iphonesentiment
pcaTest$iphonesentiment <- rcTest$iphonesentiment


# RC MODEL :  RF--------------------------------------------------------------
start_time <- Sys.time()

model.rc.pca <- train(iphonesentiment ~ .,
                           data = pcaTrain,
                           method = "rf", 
                           # tuneLength = 2,
                           tuneGrid = expand.grid(mtry = c(2, 3, 4)),
                           # tuneGrid = expand.grid(n = c(1:10)),
                           trControl = ctrl,
                           # preProc = c("pca")
)

end_time <- Sys.time()
end_time - start_time


# check performance
predict_rc_pca <- predict(model.rc.pca, pcaTest)

confusionMatrix(predict_rc_pca, pcaTest$iphonesentiment)
