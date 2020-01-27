# -------------------------------------------------------------------------
# GOAL : Main Script
# DESCRIPTION : Most important preprocessing and model
# training steps are here, as well as the source for the LargeMatrix.
# NOTE : The data appears to be a mess. variables that should not concern
# iphone sentiment (f.i. the appearance of "galaxy" on a web page) are highly 
# correlated (although negatively) with iphone sentiment. I leave these kind of
# relations out of my model because I cannot explain them.
# ALSO NOTE : The script to create dummy variables can be found under
# 'Sentiment - Dummies for DisCamPer'
# DEVELOPER : BEREND
# Mon Jan 27 11:47:35 2020 ------------------------------
# -------------------------------------------------------------------------

# Pacman -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, e1071, caret, doParallel, magrittr,
               ggplot2, corrplot, pls)

# Get Data ----------------------------------------------------------------
apple <- read.csv("Data/iphone_smallmatrix_labeled_8d.csv")

# Relevant columns only ---------------------------------------------------
# apple %<>% select(ios, samsunggalaxy, starts_with("iphone"))

# Filter out 'iphone' = 0 -------------------------------------------------
apple %<>% filter(iphone != 0)

# Feature engineering : recode() the dependent variable -------------------
# recode sentiment to combine factor levels (0, 1, 2), (3, 4)  and 5
apple$iphonesentiment <- recode(apple$iphonesentiment,
                                "0" = 1, "1" = 1,
                                "2" = 2, "3" = 2,
                                "4" = 3, "5" = 3) 

# sentiment as ordered factor
apple$iphonesentiment %<>% factor(ordered = T)


# Train Model --------------------------------------------------------------
# modify resampling method
ctrl <- trainControl(method = "repeatedcv",
                     verboseIter = TRUE,
                     repeats = 2,
                     number = 4
)

# train model
start_time <- Sys.time()

model <- train(iphonesentiment ~ .,
               data = apple,
               method = "rf",
               # tuneLength = 2,
               tuneGrid = expand.grid(mtry = c(2, 3, 4)),
               # tuneGrid = expand.grid(n = c(1:10)),
               trControl = ctrl,
               preProc = c("center", "scale")
)

end_time <- Sys.time()
end_time - start_time


# Inspect Performance -----------------------------------------------------
plot(model)

plot(varImp(model))







# Run on LargeMatrix ------------------------------------------------------
# load Data ---------------------------------------------------------------
myMatrix <- read.csv2(file = "Data/AWS cluster in parts/LargeMatrix.csv",
                      sep = ",")

# Introduce column 'iphonesentiment' --------------------------------------
myMatrix[ , "iphonesentiment"] <- NA
myMatrix$iphonesentiment %<>% factor(ordered = TRUE, levels = c("1", "2", "3"))

# Relevant columns only ---------------------------------------------------
relevant_columns <- apple %>% colnames()
myMatrix <- myMatrix[ , names(myMatrix) %in% relevant_columns]

# Run predictions ---------------------------------------------------------
sentiments <- predict(model, myMatrix)
# check distribution
summary(sentiments)






