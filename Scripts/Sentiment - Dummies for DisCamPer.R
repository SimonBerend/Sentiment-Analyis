# -------------------------------------------------------------------------
# GOAL : Create Dummy Variables
# DESCRIPTION : Here, I make dummy variables that
# summarize the relation between pos/neg/unc. 
# DEVELOPER : BEREND
# Mon Jan 27 14:01:39 2020 ------------------------------
# -------------------------------------------------------------------------

# Pacman -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, magrittr, corrplot)

# Get Data ----------------------------------------------------------------
apple <- read.csv("Data/iphone_smallmatrix_labeled_8d.csv")

# Relevant columns only ---------------------------------------------------
apple %<>% select(starts_with("iphone"))

# mutate colums to dummies
# display
apple %<>% 
  mutate(display = ifelse(iphonedispos > iphonedisneg & iphonedispos > iphonedisunc, 1, 0))
#                          ifelse (iphonedisneg > iphonedispos & iphonedisneg > iphonedisunc, 0,
 #                                 5)))
# camera
apple %<>% 
  mutate(camera = ifelse(iphonecampos > iphonecamneg & iphonecampos > iphonecamunc, 1, 0))
#                          ifelse (iphonecamneg > iphonecampos & iphonecamneg > iphonecamunc, 0,
 #                                 5)))
# performance
apple %<>% 
  mutate(performance = ifelse(iphoneperpos > iphoneperneg & iphoneperpos > iphoneperunc, 1, 0))
#                          ifelse (iphoneperneg > iphoneperpos & iphoneperneg > iphoneperunc, 0,
 #                                 5)))

# select new variables
apple %<>% select(iphone, display, camera, performance, iphonesentiment)

# set variable as ordered factors
apple$display %<>% factor()      #ordered = T, levels = c("0", "5", "10"))
apple$camera %<>% factor()        #ordered = T, levels = c("0", "5", "10"))
apple$performance %<>% factor()   #ordered = T, levels = c("0", "5", "10"))
