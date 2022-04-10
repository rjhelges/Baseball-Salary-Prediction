library(tidyverse)
library(tidymodels)
library(corrplot)
library(tibble)
library(vip)

rm(list = ls())

load(file = "data/pitcher_dta.rda")

pitcher_dta <- pitcher_dta %>%
  mutate(LOB_rate_C = as.numeric(str_replace(LOB_rate_C, "%", "")),
         LOB_rate_P1 = as.numeric(str_replace(LOB_rate_P1, "%", "")),
         LOB_rate_P2 = as.numeric(str_replace(LOB_rate_P2, "%", "")),
         GB_rate_C = as.numeric(str_replace(GB_rate_C, "%", "")),
         GB_rate_P1 = as.numeric(str_replace(GB_rate_P1, "%", "")),
         GB_rate_P2 = as.numeric(str_replace(GB_rate_P2, "%", "")),
         HR_FB_rate_C = as.numeric(str_replace(HR_FB_rate_C, "%", "")),
         HR_FB_rate_P1 = as.numeric(str_replace(HR_FB_rate_P1, "%", "")),
         HR_FB_rate_P2 = as.numeric(str_replace(HR_FB_rate_P2, "%", ""))) %>%
  replace(is.na(.), 0)

pitcher <- as.tibble(pitcher_dta)

multi_metric <- metric_set(mae, mape)

corrplot(cor(pitcher[,c(2:73)]), type = "upper")

# corr_cols <- c('Salary_Y', 'Age_C', 'PA_C', 'HR_C', 'R_C', 'RBI_C', 'WAR_C', 'MLS_C', 'Salary_C',
#                'PA_P1', 'HR_P1', 'R_P1', 'RBI_P1', 'WAR_P1', 'MLS_P1', 'Salary_P1',
#                'PA_P2', 'HR_P2', 'R_P2', 'RBI_P2', 'WAR_P2', 'MLS_P2', 'Salary_P2')
# 
# corrplot(cor(pitcher_dta[,(colnames(pitcher_dta) %in% corr_cols)], use="pairwise.complete.obs"), type = "upper")

set.seed(333)

data_split <- initial_split(pitcher, prop = .75)

train_data <- training(data_split)
test_data <- testing(data_split)

pitcher_rec_stand <- 
  recipe(Salary_Y ~ ., data = train_data) %>%
  update_role(Player, new_role = "ID") %>%
  step_normalize(all_numeric(), -all_outcomes())

reg_mod <-
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

reg_grid <- grid_latin_hypercube(
  penalty(),
  mixture(),
  size = 30
)

set.seed(333)
folds <- vfold_cv(train_data, 12)

set.seed(333)
reg_wf <- workflow() %>%
  add_model(reg_mod) %>%
  add_recipe(pitcher_rec_stand)

reg_res <-
  reg_wf %>%
  tune_grid(
    resamples = folds,
    grid = reg_grid,
    metrics = multi_metric
  )

best_reg <- reg_res %>% select_best("mape")

final_reg_p <- reg_wf %>%
  finalize_workflow(best_reg) %>%
  fit(train_data)

reg_coefs <- final_reg_p %>% 
  extract_fit_parsnip() %>% 
  tidy() %>%
  mutate(coef_magnitude = abs(estimate)) %>%
  select(c('term', 'estimate', 'coef_magnitude'))

final_fit_reg <- final_reg_p %>%
  last_fit(data_split)

reg_preds <- final_fit_reg$.predictions

mae(test_data, Salary_Y, reg_preds[[1]]$.pred)
mape(test_data, Salary_Y, reg_preds[[1]]$.pred)

mae(test_data, Salary_Y, Salary_C)
mape(test_data, Salary_Y, Salary_C)


reg_pred_table_p <- test_data %>%
  mutate(Salary_Pred = reg_preds[[1]]$.pred,
         Per_error = (reg_preds[[1]]$.pred - Salary_Y) / Salary_Y,
         Salary_C = test_data$Salary_C) %>%
  select(c(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred, Per_error))
# 
# ggplot(reg_pred_table, aes(MLS_C, Per_error)) + 
#   geom_point()
# 
# ggplot(reg_pred_table, aes(Salary_Y, Per_error)) + 
#   geom_point(aes(color = MLS_C))
# 
# ggplot(reg_pred_table, aes(Salary_Y, Salary_Pred)) + 
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# rmse(test_data, Salary_Y, Salary_C)
# rsq(test_data, Salary_Y, Salary_C)
# mean(abs((test_data$Salary_C - test_data$Salary_Y)) / test_data$Salary_Y)
# mae(test_data, Salary_Y, Salary_C)
# 
# baseline_results <- c("Baseline", rmse(test_data, Salary_Y, Salary_C)$.estimate, rsq(test_data, Salary_Y, Salary_C)$.estimate,
#                       mae(test_data, Salary_Y, Salary_C)$.estimate, mean(abs((test_data$Salary_C - test_data$Salary_Y)) / test_data$Salary_Y))
# 
# full_reg_results <- c("Full Reg", rmse(test_data, Salary_Y, reg_preds[[1]]$.pred)$.estimate, rsq(test_data, Salary_Y, reg_preds[[1]]$.pred)$.estimate,
#                       mae(test_data, Salary_Y, reg_preds[[1]]$.pred)$.estimate, 
#                       mean(abs((reg_preds[[1]]$.pred - test_data$Salary_Y)) / test_data$Salary_Y))


## ---------------------------- Boost -------------------------------------- ##

pitcher_rec <- 
  recipe(Salary_Y ~ ., data = train_data) %>%
  update_role(Player, new_role = "ID")

tune_boost <-
  boost_tree(
    tree_depth = tune(),
    trees = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    learn_rate = tune(),
    mtry = tune()
  ) %>%
  set_engine("xgboost", nthread = 8) %>%
  set_mode("regression")


boost_grid <- grid_latin_hypercube(
  trees(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(c(0.2, 0.9)),
  mtry(range = c(10,70)),
  learn_rate(),
  size = 100
)

set.seed(333)
boost_wf <- workflow() %>%
  add_model(tune_boost) %>%
  add_recipe(pitcher_rec)

boost_res <- boost_wf %>%
  tune_grid(
    resamples = folds,
    grid = boost_grid,
    metrics = multi_metric
  )

best_boost <- boost_res %>% select_best("mape")

final_boost_p <- boost_wf %>%
  finalize_workflow(best_boost) %>%
  fit(train_data)

boost_preds <- predict(final_boost_p, test_data)

mae(test_data, Salary_Y, boost_preds$.pred)
mape(test_data, Salary_Y, boost_preds$.pred)

boost_pred_table_p <- test_data %>%
  mutate(Salary_Pred = boost_preds$.pred,
         Salary_diff = boost_preds$.pred - Salary_Y,
         Per_error = (boost_preds$.pred - Salary_Y) / Salary_Y,
         Salary_C = test_data$Salary_C) %>%
  select(c(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred, Salary_diff, Per_error))

# 
# ggplot(boost_pred_table, aes(MLS_C, Per_error)) + 
#   geom_point()
# 
# ggplot(boost_pred_table, aes(MLS_C, Salary_diff)) + 
#   geom_point()
# 
# ggplot(boost_pred_table, aes(Salary_Y, Per_error)) + 
#   geom_point(aes(color = MLS_C))
# 
# ggplot(boost_pred_table, aes(Salary_Y, Salary_Pred)) + 
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# ggplot(test_data, aes(Salary_Y, Salary_C)) + 
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# full_boost_results <- c("Full Boost", rmse(test_data, Salary_Y, boost_preds$.pred)$.estimate, rsq(test_data, Salary_Y, boost_preds$.pred)$.estimate,
#                       mae(test_data, Salary_Y, boost_preds$.pred)$.estimate, 
#                       mean(abs((boost_preds$.pred - test_data$Salary_Y)) / test_data$Salary_Y))

## ------------------------------ RF --------------------------------------- ##

tune_rf <-
  rand_forest(
    trees = tune(),
    min_n = tune(),
    mtry = tune()
  ) %>%
  set_engine("ranger", importance = "impurity", num.threads = 8) %>%
  set_mode("regression")


rf_grid <- grid_latin_hypercube(
  trees(),
  min_n(),
  mtry(range = c(10,70)),
  size = 100
)

set.seed(333)
rf_wf <- workflow() %>%
  add_model(tune_rf) %>%
  add_recipe(pitcher_rec)

rf_res <- rf_wf %>%
  tune_grid(
    resamples = folds,
    grid = rf_grid,
    metrics = multi_metric
  )

best_rf <- rf_res %>% select_best("mape")

final_rf_p <- rf_wf %>%
  finalize_workflow(best_rf) %>%
  fit(train_data)

rf_preds <- predict(final_rf_p, test_data)

mae(test_data, Salary_Y, rf_preds$.pred)
mape(test_data, Salary_Y, rf_preds$.pred)

rf_pred_table_p <- test_data %>%
  mutate(Salary_Pred = rf_preds$.pred,
         Salary_diff = rf_preds$.pred - Salary_Y,
         Per_error = (rf_preds$.pred - Salary_Y) / Salary_Y,
         Salary_C = test_data$Salary_C) %>%
  select(c(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred, Salary_diff, Per_error))

# final_rf %>%
#     pull_workflow_fit() %>%
#     vip(geom = "point")
# 
# final_boost %>%
#   pull_workflow_fit() %>%
#   vip(geom = "point")

# 
# ggplot(rf_pred_table, aes(MLS_C, Per_error)) + 
#   geom_point()
# 
# ggplot(rf_pred_table, aes(MLS_C, Salary_diff)) + 
#   geom_point()
# 
# ggplot(rf_pred_table, aes(Salary_Y, Per_error)) + 
#   geom_point(aes(color = MLS_C))
# 
# ggplot(rf_pred_table, aes(Salary_Y, Salary_Pred)) + 
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# sd(rf_preds$.pred - test_data$Salary_Y)
# sd(boost_preds$.pred - test_data$Salary_Y)
# sd(reg_preds[[1]]$.pred - test_data$Salary_Y)
# 
# ggplot(rf_pred_table, aes(Per_error)) + geom_histogram()
# ggplot(reg_pred_table, aes(Per_error)) + geom_histogram()
# 
# full_rf_results <- c("Full RF", rmse(test_data, Salary_Y, rf_preds$.pred)$.estimate, rsq(test_data, Salary_Y, rf_preds$.pred)$.estimate,
#                         mae(test_data, Salary_Y, rf_preds$.pred)$.estimate, 
#                         mean(abs((rf_preds$.pred - test_data$Salary_Y)) / test_data$Salary_Y))
# 
# results_table <- rbind(baseline_results, full_reg_results, full_boost_results, full_rf_results)

### ------------------------------------------------------------------###
##            Split model by MLS                                       ##
### ------------------------------------------------------------------###

pitcher_dta_c <- pitcher_dta %>%
  select(c(Player, Age_C, IP_C, W_C, GS_C, FIP_minus_C, WAR_C, ERA_minus_C, MLS_C, Salary_C,
           IP_P1, W_P1, GS_P1, FIP_minus_P1, WAR_P1, ERA_minus_P1, Salary_P1,
           IP_P2, W_P2, GS_P2, FIP_minus_P2, WAR_P2, ERA_minus_P2, Salary_P2,
           Salary_Y))

pitcher_c <- as.tibble(pitcher_dta_c)

set.seed(333)

data_split_c <- initial_split(pitcher_c, prop = .75)

train_data_c <- training(data_split_c)
test_data_c <- testing(data_split_c)

pitcher_rec_stand_c <- 
  recipe(Salary_Y ~ ., data = train_data_c) %>%
  update_role(Player, new_role = "ID") %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_interact(terms = ~ Salary_C:all_predictors())

reg_mod_c <-
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

reg_grid_c <- grid_latin_hypercube(
  penalty(),
  mixture(),
  size = 30
)

set.seed(333)
folds_c <- vfold_cv(train_data_c, 12)

set.seed(333)
reg_wf_c <- workflow() %>%
  add_model(reg_mod_c) %>%
  add_recipe(pitcher_rec_stand_c)

reg_res_c <-
  reg_wf_c %>%
  tune_grid(
    resamples = folds_c,
    grid = reg_grid_c,
    metrics = multi_metric
  )

best_reg_c <- reg_res_c %>% select_best("mape")

final_reg_p_c <- reg_wf_c %>%
  finalize_workflow(best_reg_c) %>%
  fit(train_data_c)

reg_coefs_c <- final_reg_p_c %>% 
  extract_fit_parsnip() %>% 
  tidy() %>%
  mutate(coef_magnitude = abs(estimate)) %>%
  select(c('term', 'estimate', 'coef_magnitude'))

final_fit_reg_c <- final_reg_p_c %>%
  last_fit(data_split_c)

reg_preds_c <- final_fit_reg_c$.predictions

mae(test_data_c, Salary_Y, reg_preds_c[[1]]$.pred)
mape(test_data_c, Salary_Y, reg_preds_c[[1]]$.pred)

mae(test_data_c, Salary_Y, Salary_C)
mape(test_data_c, Salary_Y, Salary_C)


reg_pred_table_p_c <- test_data_c %>%
  mutate(Salary_Pred = reg_preds_c[[1]]$.pred,
         Per_error = (reg_preds_c[[1]]$.pred - Salary_Y) / Salary_Y,
         Salary_C = test_data$Salary_C) %>%
  select(c(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred, Per_error))

# ggplot(reg_pred_table_c, aes(MLS_C, Per_error)) +
#   geom_point()
# 
# ggplot(reg_pred_table_c, aes(Salary_Y, Per_error)) +
#   geom_point(aes(color = MLS_C))
# 
# ggplot(reg_pred_table_c, aes(Salary_Y, Salary_Pred)) +
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# c_reg_results <- c("Curr Reg", rmse(test_data_c, Salary_Y, reg_preds_c[[1]]$.pred)$.estimate, rsq(test_data_c, Salary_Y, reg_preds_c[[1]]$.pred)$.estimate,
#                       mae(test_data_c, Salary_Y, reg_preds_c[[1]]$.pred)$.estimate, 
#                       mean(abs((reg_preds_c[[1]]$.pred - test_data_c$Salary_Y)) / test_data_c$Salary_Y))

## ---------------------------- Boost Current ----------------------------- ##

pitcher_rec_c <- 
  recipe(Salary_Y ~ ., data = train_data) %>%
  update_role(Player, new_role = "ID") %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_interact(terms = ~ Salary_C:all_predictors())

set.seed(333)
boost_wf_c <- workflow() %>%
  add_model(tune_boost) %>%
  add_recipe(pitcher_rec_c)

boost_grid_c <- grid_latin_hypercube(
  trees(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(c(0.2, 0.9)),
  mtry(range = c(10, 140)),
  learn_rate(),
  size = 100
)


boost_res_c <- boost_wf_c %>%
  tune_grid(
    resamples = folds,
    grid = boost_grid_c,
    metrics = multi_metric
  )

best_boost_c <- boost_res_c %>% select_best("mape")

set.seed(333)
final_boost_p_c <- boost_wf_c %>%
  finalize_workflow(best_boost_c) %>%
  fit(train_data)

boost_preds_c <- predict(final_boost_p_c, test_data)

mae(test_data, Salary_Y, boost_preds_c$.pred)
mape(test_data, Salary_Y, boost_preds_c$.pred)


boost_pred_table_p_c <- test_data %>%
  mutate(Salary_Pred = boost_preds_c$.pred,
         Salary_diff = boost_preds_c$.pred - Salary_Y,
         Per_error = (boost_preds_c$.pred - Salary_Y) / Salary_Y,
         Salary_C = test_data$Salary_C) %>%
  select(c(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred, Salary_diff, Per_error))
# 
# ggplot(boost_pred_table_c, aes(MLS_C, Per_error)) + 
#   geom_point()
# 
# ggplot(boost_pred_table_c, aes(MLS_C, Salary_diff)) + 
#   geom_point()
# 
# ggplot(boost_pred_table_c, aes(Salary_Y, Per_error)) + 
#   geom_point(aes(color = MLS_C))
# 
# ggplot(boost_pred_table_c, aes(Salary_Y, Salary_Pred)) + 
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# c_boost_results <- c("Curr Boost", rmse(test_data_c, Salary_Y, boost_preds_c$.pred)$.estimate, rsq(test_data_c, Salary_Y, boost_preds_c$.pred)$.estimate,
#                         mae(test_data_c, Salary_Y, boost_preds_c$.pred)$.estimate, 
#                         mean(abs((boost_preds_c$.pred - test_data_c$Salary_Y)) / test_data_c$Salary_Y))

## ------------------------------ RF --------------------------------------- ##

tune_rf <-
  rand_forest(
    trees = tune(),
    min_n = tune(),
    mtry = tune()
  ) %>%
  set_engine("ranger", importance = "impurity", num.threads = 8) %>%
  set_mode("regression")


rf_grid_c <- grid_latin_hypercube(
  trees(),
  min_n(),
  mtry(range = c(10, 140)),
  size = 100
)

set.seed(333)
rf_wf_c <- workflow() %>%
  add_model(tune_rf) %>%
  add_recipe(pitcher_rec_c)

rf_res_c <- rf_wf_c %>%
  tune_grid(
    resamples = folds,
    grid = rf_grid_c,
    metrics = multi_metric
  )

best_rf_c <- rf_res_c %>% select_best("mape")

set.seed(333)
final_rf_p_c <- rf_wf_c %>%
  finalize_workflow(best_rf_c) %>%
  fit(train_data)

rf_preds_c <- predict(final_rf_p_c, test_data)

mae(test_data, Salary_Y, rf_preds_c$.pred)
mape(test_data, Salary_Y, rf_preds_c$.pred)

rf_pred_table_p_c <- test_data_c %>%
  mutate(Salary_Pred = rf_preds_c$.pred,
         Salary_diff = rf_preds_c$.pred - Salary_Y,
         Per_error = (rf_preds_c$.pred - Salary_Y) / Salary_Y,
         Salary_C = test_data$Salary_C) %>%
  select(c(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred, Salary_diff, Per_error))
# 
# ggplot(rf_pred_table_c, aes(MLS_C, Per_error)) + 
#   geom_point()
# 
# ggplot(rf_pred_table_c, aes(MLS_C, Salary_diff)) + 
#   geom_point()
# 
# ggplot(rf_pred_table_c, aes(Salary_Y, Per_error)) + 
#   geom_point(aes(color = MLS_C))
# 
# ggplot(rf_pred_table_c, aes(Salary_Y, Salary_Pred)) + 
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# ggplot(rf_pred_table_c, aes(Salary_diff)) + geom_histogram()

save(list = c('reg_pred_table_p', 'boost_pred_table_p', 'rf_pred_table_p', 'final_reg_p', 'final_boost_p', 'final_rf_p',
              'reg_pred_table_p_c', 'boost_pred_table_p_c', 'rf_pred_table_p_c', 'final_reg_p_c', 'final_boost_p_c', 'final_rf_p_c'), 
     file = "data/01-pitcher-analysis.rda")
