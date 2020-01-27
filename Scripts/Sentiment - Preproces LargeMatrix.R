# -------------------------------------------------------------------------
# GOAL : preproces the Large Matrix from AWS to run prediction
# DESCRIPTION : 
# DEVELOPER : BEREND
# Mon Jan 27 10:43:06 2020 ------------------------------
# -------------------------------------------------------------------------


# load Data ---------------------------------------------------------------
myMatrix <- read.csv2(file = "Data/AWS cluster in parts/LargeMatrix.csv",
                      sep = ",")


# Introduce column 'iphonesentiment' --------------------------------------
myMatrix[ , "iphonesentiment"] <- NA
myMatrix$iphonesentiment %<>% factor(ordered = TRUE, levels = c("1", "2", "3", "4"))


# Relevant columns only ---------------------------------------------------
relevant_columns <- appleRC %>% colnames()
myMatrix <- myMatrix[ , names(myMatrix) %in% relevant_columns]


# Run predictions ---------------------------------------------------------
sentiments <- predict(model, myMatrix)
# check distribution
summary(sentiments)


