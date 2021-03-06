library(tidyverse)

rm(list = ls())

pitcher_dta <- read.csv("data/pitchers_performance.csv", header = TRUE)

colnames(pitcher_dta)[1] <- "Season"

pitchers_12 <- pitcher_dta %>% filter(Season == '2012')
pitchers_13 <- pitcher_dta %>% filter(Season == '2013')
pitchers_14 <- pitcher_dta %>% filter(Season == '2014')
pitchers_15 <- pitcher_dta %>% filter(Season == '2015')
pitchers_16 <- pitcher_dta %>% filter(Season == '2016')
pitchers_17 <- pitcher_dta %>% filter(Season == '2017')
pitchers_18 <- pitcher_dta %>% filter(Season == '2018')
pitchers_19 <- pitcher_dta %>% filter(Season == '2019')

## Read in salary data ##
salaries_2012 <- read.csv("data/salaries_2012.csv", header = TRUE)
salaries_2013 <- read.csv("data/salaries_2013.csv", header = TRUE)
salaries_2014 <- read.csv("data/salaries_2014.csv", header = TRUE)
salaries_2015 <- read.csv("data/salaries_2015.csv", header = TRUE)
salaries_2016 <- read.csv("data/salaries_2016.csv", header = TRUE)
salaries_2017 <- read.csv("data/salaries_2017.csv", header = TRUE)
salaries_2018 <- read.csv("data/salaries_2018.csv", header = TRUE)
salaries_2019 <- read.csv("data/salaries_2019.csv", header = TRUE)
salaries_2020 <- read.csv("data/salaries_2020.csv", header = TRUE)

## Clean up 2018 salaries ##
salaries_2018 <- salaries_2018 %>%
  mutate(Salary = MLS,
         MLS = Pos.n,
         Pos.n = Player,
         Player = rownames(salaries_2018)) %>%
  select(-c(X))

row.names(salaries_2018) <- NULL

## Clean salary data and join with performance data##
salaries_2012 <- salaries_2012 %>%
  separate(Player, c("Last", "First"), ", ") %>%
  mutate(Last = trimws(str_replace(Last, pattern = c(" Jr."), "")),
         First = trimws(First),
         Salary = parse_number(Salary),
         Season = 2012,
         Pos = toupper(Pos)) %>%
  # separate(Last, c("Last", "Suffix"), " ", extra = "merge") %>%
  filter(grepl("RHP|LHP", Pos)) %>%
  select(c(First, Last, MLS, Salary, Season))

pitchers_12 <- pitchers_12 %>%
  separate(Name, c("First", "Last"), " ") %>%
  mutate(Last = trimws(Last),
         First = trimws(First))

pitchers_2012_dta <- pitchers_12 %>%
  inner_join(salaries_2012, by = c('Last', 'First', 'Season'))

## 2013 ##
salaries_2013 <- salaries_2013 %>%
  separate(Player, c("Last", "First"), ", ") %>%
  mutate(Last = trimws(str_replace(Last, pattern = c(" Jr."), "")),
         First = trimws(First),
         Salary = parse_number(Salary),
         Season = 2013,
         Pos = toupper(Pos)) %>%
  # separate(Last, c("Last", "Suffix"), " ", extra = "merge") %>%
  filter(grepl("RHP|LHP", Pos)) %>%
  select(c(First, Last, MLS, Salary, Season))

pitchers_13 <- pitchers_13 %>%
  separate(Name, c("First", "Last"), " ") %>%
  mutate(Last = trimws(Last),
         First = trimws(First))

pitchers_2013_dta <- pitchers_13 %>%
  inner_join(salaries_2013, by = c('Last', 'First', 'Season'))

## 2014 ##
salaries_2014 <- salaries_2014 %>%
  separate(Player, c("Last", "First"), ", ") %>%
  mutate(Last = trimws(str_replace(Last, pattern = c(" Jr."), "")),
         First = trimws(First),
         Salary = parse_number(Salary),
         Season = 2014,
         Pos = toupper(Pos)) %>%
  # separate(Last, c("Last", "Suffix"), " ", extra = "merge") %>%
  filter(grepl("RHP|LHP", Pos)) %>%
  select(c(First, Last, MLS, Salary, Season))

pitchers_14 <- pitchers_14 %>%
  separate(Name, c("First", "Last"), " ") %>%
  mutate(Last = trimws(Last),
         First = trimws(First))

pitchers_2014_dta <- pitchers_14 %>%
  inner_join(salaries_2014, by = c('Last', 'First', 'Season'))

## 2015 ##
salaries_2015 <- salaries_2015 %>%
  separate(Player, c("Last", "First"), ", ") %>%
  mutate(Last = trimws(str_replace(Last, pattern = c(" Jr."), "")),
         First = trimws(First),
         Salary = parse_number(Salary),
         Season = 2015,
         Pos.n = toupper(Pos.n)) %>%
  # separate(Last, c("Last", "Suffix"), " ", extra = "merge") %>%
  filter(grepl("RHP|LHP", Pos.n)) %>%
  select(c(First, Last, MLS, Salary, Season))

pitchers_15 <- pitchers_15 %>%
  separate(Name, c("First", "Last"), " ") %>%
  mutate(Last = trimws(Last),
         First = trimws(First))

pitchers_2015_dta <- pitchers_15 %>%
  inner_join(salaries_2015, by = c('Last', 'First', 'Season'))

## 2016 ##
salaries_2016 <- salaries_2016 %>%
  separate(Player, c("Last", "First"), ", ") %>%
  mutate(Last = trimws(str_replace(Last, pattern = c(" Jr."), "")),
         First = trimws(First),
         Salary = parse_number(Salary),
         Season = 2016,
         Pos.n = toupper(Pos.n)) %>%
  # separate(Last, c("Last", "Suffix"), " ", extra = "merge") %>%
  filter(grepl("RHP|LHP", Pos.n)) %>%
  select(c(First, Last, MLS, Salary, Season))

pitchers_16 <- pitchers_16 %>%
  separate(Name, c("First", "Last"), " ") %>%
  mutate(Last = trimws(Last),
         First = trimws(First))

pitchers_2016_dta <- pitchers_16 %>%
  inner_join(salaries_2016, by = c('Last', 'First', 'Season'))

## 2017 ##
salaries_2017 <- salaries_2017 %>%
  separate(Player, c("Last", "First"), ", ") %>%
  mutate(Last = trimws(str_replace(Last, pattern = c(" Jr."), "")),
         First = trimws(First),
         Salary = parse_number(Salary),
         Season = 2017,
         Pos.n = toupper(Pos.n)) %>%
  # separate(Last, c("Last", "Suffix"), " ", extra = "merge") %>%
  filter(grepl("RHP|LHP", Pos.n)) %>%
  select(c(First, Last, MLS, Salary, Season))

pitchers_17 <- pitchers_17 %>%
  separate(Name, c("First", "Last"), " ") %>%
  mutate(Last = trimws(Last),
         First = trimws(First))

pitchers_2017_dta <- pitchers_17 %>%
  inner_join(salaries_2017, by = c('Last', 'First', 'Season'))

## 2018 ##
salaries_2018 <- salaries_2018 %>%
  separate(Player, c("Last", "First"), ", ") %>%
  mutate(Last = trimws(str_replace(Last, pattern = c(" Jr."), "")),
         First = trimws(First),
         Salary = parse_number(Salary),
         Season = 2018,
         Pos.n = toupper(Pos.n)) %>%
  # separate(Last, c("Last", "Suffix"), " ", extra = "merge") %>%
  filter(grepl("RHP|LHP", Pos.n)) %>%
  select(c(First, Last, MLS, Salary, Season))

pitchers_18 <- pitchers_18 %>%
  separate(Name, c("First", "Last"), " ") %>%
  mutate(Last = trimws(Last),
         First = trimws(First)) 

pitchers_2018_dta <- pitchers_18 %>%
  inner_join(salaries_2018, by = c('Last', 'First', 'Season'))

## 2019 ##
salaries_2019 <- salaries_2019 %>%
  separate(Player, c("Last", "First"), ", ") %>%
  mutate(Last = trimws(str_replace(Last, pattern = c(" Jr."), "")),
         First = trimws(First),
         Salary = parse_number(Salary),
         Season = 2019,
         Pos.n = toupper(Pos.n)) %>%
  # separate(Last, c("Last", "Suffix"), " ", extra = "merge") %>%
  filter(grepl("RHP|LHP", Pos.n)) %>%
  select(c(First, Last, MLS, Salary, Season))

pitchers_19 <- pitchers_19 %>%
  separate(Name, c("First", "Last"), " ") %>%
  mutate(Last = trimws(Last),
         First = trimws(First))

pitchers_2019_dta <- pitchers_19 %>%
  inner_join(salaries_2019, by = c('Last', 'First', 'Season'))

## 2020 ##
salaries_2020 <- salaries_2020 %>%
  separate(Player, c("Last", "First"), ", ") %>%
  mutate(Last = trimws(str_replace(Last, pattern = c(" Jr."), "")),
         First = trimws(First),
         Salary = parse_number(Salary),
         Season = 2020,
         Pos.n = toupper(Pos.n)) %>%
  # separate(Last, c("Last", "Suffix"), " ", extra = "merge") %>%
  filter(grepl("RHP|LHP", Pos.n)) %>%
  select(c(First, Last, Salary, Season))


#--------------------------------------------------------------------------------------------------#

## 2014 pitcher data with past 2 years performance
full_pitchers_2014 <- pitchers_2014_dta %>% 
  left_join(pitchers_2013_dta, by = 'playerid', suffix = c("_C", "_P1"))

colnames(pitchers_2012_dta) <- paste(colnames(pitchers_2012_dta), "P2", sep = "_")

full_pitchers_2014 <- full_pitchers_2014 %>%
  left_join(pitchers_2012_dta, by = c('playerid' = 'playerid_P2')) %>%
  select(-c(First_P1, First_P2, Last_P1, Last_P2))

## 2015 pitcher data with past 2 years performance
full_pitchers_2015 <- pitchers_2015_dta %>% 
  left_join(pitchers_2014_dta, by = 'playerid', suffix = c("_C", "_P1"))

colnames(pitchers_2013_dta) <- paste(colnames(pitchers_2013_dta), "P2", sep = "_")

full_pitchers_2015 <- full_pitchers_2015 %>%
  left_join(pitchers_2013_dta, by = c('playerid' = 'playerid_P2')) %>%
  select(-c(First_P1, First_P2, Last_P1, Last_P2))

## 2016 pitcher data with past 2 years performance
full_pitchers_2016 <- pitchers_2016_dta %>% 
  left_join(pitchers_2015_dta, by = 'playerid', suffix = c("_C", "_P1"))

colnames(pitchers_2014_dta) <- paste(colnames(pitchers_2014_dta), "P2", sep = "_")

full_pitchers_2016 <- full_pitchers_2016 %>%
  left_join(pitchers_2014_dta, by = c('playerid' = 'playerid_P2')) %>%
  select(-c(First_P1, First_P2, Last_P1, Last_P2))

## 2017 pitcher data with past 2 years performance
full_pitchers_2017 <- pitchers_2017_dta %>% 
  left_join(pitchers_2016_dta, by = 'playerid', suffix = c("_C", "_P1"))

colnames(pitchers_2015_dta) <- paste(colnames(pitchers_2015_dta), "P2", sep = "_")

full_pitchers_2017 <- full_pitchers_2017 %>%
  left_join(pitchers_2015_dta, by = c('playerid' = 'playerid_P2')) %>%
  select(-c(First_P1, First_P2, Last_P1, Last_P2))

## 2018 pitcher data with past 2 years performance
full_pitchers_2018 <- pitchers_2018_dta %>% 
  left_join(pitchers_2017_dta, by = 'playerid', suffix = c("_C", "_P1"))

colnames(pitchers_2016_dta) <- paste(colnames(pitchers_2016_dta), "P2", sep = "_")

full_pitchers_2018 <- full_pitchers_2018 %>%
  left_join(pitchers_2016_dta, by = c('playerid' = 'playerid_P2')) %>%
  select(-c(First_P1, First_P2, Last_P1, Last_P2))

## 2019 pitcher data with past 2 years performance
full_pitchers_2019 <- pitchers_2019_dta %>% 
  left_join(pitchers_2018_dta, by = 'playerid', suffix = c("_C", "_P1"))

colnames(pitchers_2017_dta) <- paste(colnames(pitchers_2017_dta), "P2", sep = "_")

full_pitchers_2019 <- full_pitchers_2019 %>%
  left_join(pitchers_2017_dta, by = c('playerid' = 'playerid_P2')) %>%
  select(-c(First_P1, First_P2, Last_P1, Last_P2))

## Join predictor data set with response Salary ##
## Subtract year from salaries to do join ##
salaries_2020$Season <- salaries_2020$Season - 1
salaries_2019$Season <- salaries_2019$Season - 1
salaries_2018$Season <- salaries_2018$Season - 1
salaries_2017$Season <- salaries_2017$Season - 1
salaries_2016$Season <- salaries_2016$Season - 1
salaries_2015$Season <- salaries_2015$Season - 1
salaries_2014$Season <- salaries_2014$Season - 1
salaries_2013$Season <- salaries_2013$Season - 1

full_pitchers_2019 <- full_pitchers_2019 %>%
  inner_join(salaries_2020, by = c('Last_C' = 'Last', 'First_C' = 'First', 'Season_C' = 'Season')) %>%
  rename(Salary_Y = Salary)

full_pitchers_2018 <- full_pitchers_2018 %>%
  inner_join(salaries_2019, by = c('Last_C' = 'Last', 'First_C' = 'First', 'Season_C' = 'Season')) %>%
  rename(Salary_Y = Salary) %>%
  select(-MLS)

full_pitchers_2017 <- full_pitchers_2017 %>%
  inner_join(salaries_2018, by = c('Last_C' = 'Last', 'First_C' = 'First', 'Season_C' = 'Season')) %>%
  rename(Salary_Y = Salary) %>%
  select(-MLS)

full_pitchers_2016 <- full_pitchers_2016 %>%
  inner_join(salaries_2017, by = c('Last_C' = 'Last', 'First_C' = 'First', 'Season_C' = 'Season')) %>%
  rename(Salary_Y = Salary) %>%
  select(-MLS)

full_pitchers_2015 <- full_pitchers_2015 %>%
  inner_join(salaries_2016, by = c('Last_C' = 'Last', 'First_C' = 'First', 'Season_C' = 'Season')) %>%
  rename(Salary_Y = Salary) %>%
  select(-MLS)

full_pitchers_2014 <- full_pitchers_2014 %>%
  inner_join(salaries_2015, by = c('Last_C' = 'Last', 'First_C' = 'First', 'Season_C' = 'Season')) %>%
  rename(Salary_Y = Salary) %>%
  select(-MLS)


## Union all records together
pitcher_dta <- bind_rows(full_pitchers_2019, full_pitchers_2018, full_pitchers_2017, full_pitchers_2016, full_pitchers_2015, full_pitchers_2014)

# Remove some anomalous data
pitcher_dta <- pitcher_dta %>% filter(!(First_C == 'Miguel' & Last_C == 'Gonzalez'))
pitcher_dta <- pitcher_dta %>% filter(!(Team_C == 'LAA' & Last_C == 'Carpenter'))

pitcher_dta <- pitcher_dta %>%
  mutate(Player = paste(Season_C, First_C, Last_C, sep="_")) %>%
  select(-c(Season_C, First_C, Last_C, Team_C, playerid, Season_P1, Team_P1, Age_P1, MLS_P1, Season_P2, Team_P2, Age_P2, MLS_P2)) %>%
  relocate(Player)

save(list = 'pitcher_dta', file = 'data/pitcher_dta.rda')
  
