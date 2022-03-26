library(tidyverse)

rm(list = ls())

batter_dta <- read.csv("data/batters_performance.csv", header = TRUE)

colnames(batter_dta)[1] <- "Season"

batters_12 <- batter_dta %>% filter(Season == '2012')
batters_13 <- batter_dta %>% filter(Season == '2013')
batters_14 <- batter_dta %>% filter(Season == '2014')
batters_15 <- batter_dta %>% filter(Season == '2015')
batters_16 <- batter_dta %>% filter(Season == '2016')
batters_17 <- batter_dta %>% filter(Season == '2017')
batters_18 <- batter_dta %>% filter(Season == '2018')
batters_19 <- batter_dta %>% filter(Season == '2019')

## 2014 batter data with past 2 years performance
batters_14_full <- batters_14 %>% 
  left_join(batters_13, by = 'playerid', suffix = c("_14", "_13"))

colnames(batters_12) <- paste(colnames(batters_12), "12", sep = "_")

batters_14_full <- batters_14_full %>%
  left_join(batters_12, by = c('playerid' = 'playerid_12'))

## 2015 batter data with past 2 years performance
batters_15_full <- batters_15 %>% 
  left_join(batters_14, by = 'playerid', suffix = c("_15", "_14"))

colnames(batters_13) <- paste(colnames(batters_13), "13", sep = "_")

batters_15_full <- batters_15_full %>%
  left_join(batters_13, by = c('playerid' = 'playerid_13'))
