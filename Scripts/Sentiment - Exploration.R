
# -------------------------------------------------------------------------
# GOAL :
# DESCRIPTION :
# DEVELOPER : BEREND
# Wed Jan 22 14:57:45 2020 ------------------------------
# -------------------------------------------------------------------------

# Pacman -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, e1071, caret, doParallel,
               plotly, magrittr, ggplot2, corrplot)


# Get Data ----------------------------------------------------------------
galaxy <- read.csv("Data/galaxy_smallmatrix_labeled_8d.csv")
apple <- read.csv("Data/iphone_smallmatrix_labeled_8d.csv")


# Relevant columns only ---------------------------------------------------
# Note : disregard iOS data
galaxy %<>% select(starts_with("samsung"), contains("senti"))
apple %<>% select(starts_with("iphone"))


# Distribution of sentiments -------------------------------------------------------------------------
plot_ly(galaxy, x= ~galaxy$galaxysentiment, type='histogram', name = "galaxy") %>% 
  add_trace(data = apple, x = apple$iphonesentiment, name = "apple")


# -------------------------------------------------------------------------
cor(apple, apple$iphonesentiment)
cor(galaxy, galaxy$galaxysentiment)

cor_app <- cor(apple)
cor_gal <- cor(galaxy)

corrplot(cor_app)
corrplot(cor_gal)