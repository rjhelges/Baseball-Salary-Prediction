library(tidyverse)
library(tidymodels)
library(corrplot)
library(tibble)
library(vip)

rm(list = ls())

load(file = "data/batter_dta.rda")

batter_dta <- batter_dta %>%
  mutate(BB_rate_C = as.numeric(str_replace(BB_rate_C, "%", "")),
         BB_rate_P1 = as.numeric(str_replace(BB_rate_P1, "%", "")),
         BB_rate_P2 = as.numeric(str_replace(BB_rate_P2, "%", "")),
         K_rate_C = as.numeric(str_replace(K_rate_C, "%", "")),
         K_rate_P1 = as.numeric(str_replace(K_rate_P1, "%", "")),
         K_rate_P2 = as.numeric(str_replace(K_rate_P2, "%", ""))) %>%
  replace(is.na(.), 0)

batter <- as.tibble(batter_dta)

multi_metric <- metric_set(mae, mape)

set.seed(333)

data_split <- initial_split(batter, prop = .75)

train_data <- training(data_split)
test_data <- testing(data_split)

train_data_rook <- train_data %>% filter(MLS_C < 5)
train_data_fa <- train_data %>% filter(MLS_C >= 5)

test_data_rook <- test_data %>% filter(MLS_C < 5)
test_data_fa <- test_data %>% filter(MLS_C >= 5)

#### -------------------- Rookie Contract Models --------------------------- ####

batter_rook_stand <- 
  recipe(Salary_Y ~ ., data = train_data_rook) %>%
  update_role(Player, new_role = "ID") %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_interact(terms = ~ Salary_C:all_predictors())

reg_mod <-
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

reg_grid <- grid_regular(penalty(),
                         mixture(),
                         levels = 10)

set.seed(333)
folds_rook <- vfold_cv(train_data_rook, 10)

set.seed(333)
reg_wf <- workflow() %>%
  add_model(reg_mod) %>%
  add_recipe(batter_rook_stand)

reg_res <-
  reg_wf %>%
  tune_grid(
    resamples = folds_rook,
    grid = reg_grid,
    metrics = multi_metric
  )

best_reg <- reg_res %>% select_best("mape")

final_reg_rook <- reg_wf %>%
  finalize_workflow(best_reg) %>%
  fit(train_data_rook)

reg_coefs_rook <- final_reg_rook %>% 
  extract_fit_parsnip() %>% 
  tidy() %>%
  mutate(coef_magnitude = abs(estimate)) %>%
  select(c('term', 'estimate', 'coef_magnitude'))

reg_preds_rook <- predict(final_reg_rook, test_data_rook)

mae(test_data_rook, Salary_Y, reg_preds_rook$.pred)
mape(test_data_rook, Salary_Y, reg_preds_rook$.pred)

mae(test_data_rook, Salary_Y, Salary_C)
mape(test_data_rook, Salary_Y, Salary_C)

reg_pred_table_rook <- test_data_rook %>%
  mutate(Salary_Pred = reg_preds_rook$.pred,
         Per_error = (reg_preds_rook$.pred - Salary_Y) / Salary_Y,
         Salary_diff = reg_preds_rook$.pred - Salary_Y,
         Salary_C = test_data_rook$Salary_C) %>%
  select(c(Player, MLS_C, Salary_Y, Salary_Pred, Salary_diff, Per_error, Salary_C))

# ggplot(reg_pred_table_rook, aes(MLS_C, Per_error)) +
#   geom_point()
# 
# ggplot(reg_pred_table_rook, aes(Salary_Y, Per_error)) +
#   geom_point(aes(color = MLS_C))
# 
# ggplot(reg_pred_table_rook, aes(Salary_Y, Salary_Pred)) +
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# ggplot(reg_pred_table_rook, aes(Per_error)) + geom_histogram()

## ------------------------ RF Rookie ------------------------------------- ##

batter_rook <- 
  recipe(Salary_Y ~ ., data = train_data_rook) %>%
  update_role(Player, new_role = "ID") %>%
  step_interact(terms = ~ Salary_C:all_predictors())

tune_rf <-
  rand_forest(
    trees = 1000,
    min_n = tune(),
    mtry = tune()
  ) %>%
  set_engine("ranger", importance = "impurity", num.threads = 8) %>%
  set_mode("regression")


set.seed(333)
rf_wf_rook <- workflow() %>%
  add_model(tune_rf) %>%
  add_recipe(batter_rook)

set.seed(333)
tune_res <- tune_grid(
  rf_wf_rook,
  resamples = folds_rook,
  grid = 20
)

tune_res %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "RMSE")

rf_grid_rook <- grid_regular(min_n(range = c(2, 10)),
                             mtry(range(30, 90)),
                             levels = c(5, 7))

rf_res_rook <- rf_wf_rook %>%
  tune_grid(
    resamples = folds_rook,
    grid = rf_grid_rook,
    metrics = multi_metric
  )

best_rf_rook <- rf_res_rook %>% select_best("mape")

set.seed(333)
final_rf_rook <- rf_wf_rook %>%
  finalize_workflow(best_rf_rook) %>%
  fit(train_data_rook)

# save(list = 'final_rf_c', file = 'data/tuned_rf_model.rda')

rf_preds_rook <- predict(final_rf_rook, test_data_rook)

mae(test_data_rook, Salary_Y, rf_preds_rook$.pred)
mape(test_data_rook, Salary_Y, rf_preds_rook$.pred)

rf_pred_table_rook <- test_data_rook %>%
  mutate(Salary_Pred = rf_preds_rook$.pred,
         Per_error = (rf_preds_rook$.pred - Salary_Y) / Salary_Y,
         Salary_diff= rf_preds_rook$.pred - Salary_Y) %>%
  select(c(Player, MLS_C, Salary_Y, Salary_Pred, Salary_diff, Per_error))
# 
# ggplot(rf_pred_table_rook, aes(MLS_C, Per_error)) +
#   geom_point()
# 
# ggplot(rf_pred_table_rook, aes(Salary_Y, Per_error)) +
#   geom_point(aes(color = MLS_C))
# 
# ggplot(rf_pred_table_rook, aes(Salary_Y, Salary_Pred)) +
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# ggplot(rf_pred_table_rook, aes(Salary_diff)) + geom_histogram()
# 
# ggplot(test_data_rook, aes((Salary_C - Salary_Y)/Salary_Y)) + geom_histogram()
# 
# final_rf_rook %>%
#   pull_workflow_fit() %>%
#   vip(geom = "point")

## ------------------------ Reg FA ------------------------------------- ##

set.seed(333)
folds_fa <- vfold_cv(train_data_fa, 10)

batter_fa_stand <- 
  recipe(Salary_Y ~ ., data = train_data_fa) %>%
  update_role(Player, new_role = "ID") %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_interact(terms = ~ Salary_C:all_predictors())


set.seed(333)
reg_wf_fa <- workflow() %>%
  add_model(reg_mod) %>%
  add_recipe(batter_fa_stand)

reg_res_fa <-
  reg_wf_fa %>%
  tune_grid(
    resamples = folds_fa,
    grid = reg_grid,
    metrics = multi_metric
  )

best_reg_fa <- reg_res_fa %>% select_best("mape")

final_reg_fa <- reg_wf_fa %>%
  finalize_workflow(best_reg_fa) %>%
  fit(train_data_fa)

reg_coefs_fa <- final_reg_fa %>% 
  extract_fit_parsnip() %>% 
  tidy() %>%
  mutate(coef_magnitude = abs(estimate)) %>%
  select(c('term', 'estimate', 'coef_magnitude'))

reg_preds_fa <- predict(final_reg_fa, test_data_fa)

mae(test_data_fa, Salary_Y, reg_preds_fa$.pred)
mape(test_data_fa, Salary_Y, reg_preds_fa$.pred)

mae(test_data_fa, Salary_Y, Salary_C)
mape(test_data_fa, Salary_Y, Salary_C)

reg_pred_table_fa <- test_data_fa %>%
  mutate(Salary_Pred = reg_preds_fa$.pred,
         Per_error = (reg_preds_fa$.pred - Salary_Y) / Salary_Y,
         Salary_diff = reg_preds_fa$.pred - Salary_Y,
         Salary_C = test_data_fa$Salary_C) %>%
  select(c(Player, MLS_C, Salary_Y, Salary_Pred, Salary_diff, Per_error, Salary_C))
# 
# ggplot(reg_pred_table_fa, aes(MLS_C, Per_error)) +
#   geom_point()
# 
# ggplot(reg_pred_table_fa, aes(Salary_Y, Per_error)) +
#   geom_point(aes(color = MLS_C))
# 
# ggplot(reg_pred_table_fa, aes(Salary_Y, Salary_Pred)) +
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# ggplot(reg_pred_table_fa, aes(Per_error)) + geom_histogram()

## ------------------------ RF FA ------------------------------------- ##

batter_fa <- 
  recipe(Salary_Y ~ ., data = train_data_fa) %>%
  update_role(Player, new_role = "ID") %>%
  step_interact(terms = ~ all_predictors():all_predictors())

tune_rf <-
  rand_forest(
    trees = 1000,
    min_n = tune(),
    mtry = tune()
  ) %>%
  set_engine("ranger", importance = "impurity", num.threads = 8) %>%
  set_mode("regression")


set.seed(333)
rf_wf_fa <- workflow() %>%
  add_model(tune_rf) %>%
  add_recipe(batter_fa)

set.seed(333)
tune_res <- tune_grid(
  rf_wf_fa,
  resamples = folds_fa,
  grid = 20
)

tune_res %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "RMSE")

set.seed(333)
rf_grid_fa <- grid_random(trees(),
                          min_n(),
                          finalize(mtry(), train_data_rook),
                          size = 100)


rf_res_fa <- rf_wf_fa %>%
  tune_grid(
    resamples = folds,
    grid = rf_grid_fa,
    metrics = multi_metric
  )

best_rf_fa <- rf_res_fa %>% select_best("mape")

set.seed(333)
final_rf_fa <- rf_wf_fa %>%
  finalize_workflow(best_rf_fa) %>%
  fit(train_data_fa)

# save(list = 'final_rf_c', file = 'data/tuned_rf_model.rda')

rf_preds_fa <- predict(final_rf_fa, test_data_fa)

mae(test_data_fa, Salary_Y, rf_preds_fa$.pred)
mape(test_data_fa, Salary_Y, rf_preds_fa$.pred)

rf_pred_table_fa <- test_data_fa %>%
  mutate(Salary_Pred = rf_preds_fa$.pred,
         Per_error = (rf_preds_fa$.pred - Salary_Y) / Salary_Y,
         Salary_diff= rf_preds_fa$.pred - Salary_Y) %>%
  select(c(Player, MLS_C, Salary_Y, Salary_Pred, Salary_diff, Per_error))
# 
# ggplot(rf_pred_table_fa, aes(MLS_C, Per_error)) +
#   geom_point()
# 
# ggplot(rf_pred_table_fa, aes(Salary_Y, Per_error)) +
#   geom_point(aes(color = MLS_C))
# 
# ggplot(rf_pred_table_fa, aes(Salary_Y, Salary_Pred)) +
#   geom_point() +
#   geom_point(aes(Salary_Y, Salary_Y), color = 'red')
# 
# ggplot(rf_pred_table_fa, aes(Salary_diff)) + geom_histogram()
# 
# ggplot(test_data_fa, aes((Salary_C - Salary_Y)/Salary_Y)) + geom_histogram()
# 
# final_rf_fa %>%
#   pull_workflow_fit() %>%
#   vip(geom = "point")

combined_reg_output <- test_data %>%
  left_join(reg_pred_table_rook, by = 'Player', suffix = c(".x", "")) %>%
  select(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred) %>%
  left_join(reg_pred_table_fa, by = 'Player', suffix = c("", ".y")) %>%
  mutate(MLS_C = ifelse(is.na(MLS_C), MLS_C.y, MLS_C),
         Salary_C = ifelse(is.na(Salary_C), Salary_C.y, Salary_C),
         Salary_Y = ifelse(is.na(Salary_Y), Salary_Y.y, Salary_Y),
         Salary_Pred = ifelse(is.na(Salary_Pred), Salary_Pred.y, Salary_Pred)) %>%
  select(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred)

mae(combined_reg_output, Salary_Y, Salary_Pred)

combined_rf_output <- test_data %>%
  left_join(rf_pred_table_rook, by = 'Player', suffix = c(".x", "")) %>%
  select(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred) %>%
  left_join(rf_pred_table_fa, by = 'Player', suffix = c("", ".y")) %>%
  mutate(MLS_C = ifelse(is.na(MLS_C), MLS_C.y, MLS_C),
         Salary_C = ifelse(is.na(Salary_C), Salary_C.y, Salary_C),
         Salary_Y = ifelse(is.na(Salary_Y), Salary_Y.y, Salary_Y),
         Salary_Pred = ifelse(is.na(Salary_Pred), Salary_Pred.y, Salary_Pred)) %>%
  select(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred)

mae(combined_rf_output, Salary_Y, Salary_C)

save(list = c('combined_reg_output', 'combined_rf_output', 'final_reg_rook', 'final_reg_fa', 
              'final_rf_rook', 'final_rf_fa'), file = "data/02-batter-analysis.rda")

################################# boost #########################################

batter_rook <- 
  recipe(Salary_Y ~ ., data = train_data_rook) %>%
  update_role(Player, new_role = "ID") %>%
  step_interact(terms = ~ Salary_C:all_predictors())

tune_boost <-
  boost_tree(
    tree_depth = tune(),
    trees = 1000,
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    learn_rate = tune(),
    mtry = tune()
  ) %>%
  set_engine("xgboost", nthread = 8) %>%
  set_mode("regression")


boost_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(c(0.4, 0.8)),
  finalize(mtry(), train_data),
  learn_rate(),
  size = 50
)

set.seed(333)
boost_wf_rook <- workflow() %>%
  add_model(tune_boost) %>%
  add_recipe(batter_rook)

boost_res_rook <- boost_wf_rook %>%
  tune_grid(
    resamples = folds,
    grid = boost_grid,
    metrics = multi_metric
  )

best_boost_rook <- boost_res_rook %>% select_best("mape")

set.seed(333)
final_boost_rook <- boost_wf_rook %>%
  finalize_workflow(best_boost_rook) %>%
  fit(train_data_rook)

# save(list = 'final_rf_c', file = 'data/tuned_rf_model.rda')

boost_preds_rook <- predict(final_boost_rook, test_data_rook)

mae(test_data_rook, Salary_Y, boost_preds_rook$.pred)
mape(test_data_rook, Salary_Y, boost_preds_rook$.pred)

boost_pred_table_rook <- test_data_rook %>%
  mutate(Salary_Pred = boost_preds_rook$.pred,
         Per_error = (boost_preds_rook$.pred - Salary_Y) / Salary_Y,
         Salary_diff= boost_preds_rook$.pred - Salary_Y) %>%
  select(c(Player, MLS_C, Salary_Y, Salary_Pred, Salary_diff, Per_error))

################################# FA #########################################

batter_fa <- 
  recipe(Salary_Y ~ ., data = train_data_fa) %>%
  update_role(Player, new_role = "ID") %>%
  step_interact(terms = ~ Salary_C:all_predictors())

set.seed(333)
boost_wf_fa <- workflow() %>%
  add_model(tune_boost) %>%
  add_recipe(batter_fa)

boost_res_fa <- boost_wf_fa %>%
  tune_grid(
    resamples = folds,
    grid = boost_grid,
    metrics = multi_metric
  )

best_boost_fa <- boost_res_fa %>% select_best("mape")

set.seed(333)
final_boost_fa <- boost_wf_fa %>%
  finalize_workflow(best_boost_fa) %>%
  fit(train_data_fa)

# save(list = 'final_rf_c', file = 'data/tuned_rf_model.rda')

boost_preds_fa <- predict(final_boost_fa, test_data_fa)

mae(test_data_fa, Salary_Y, boost_preds_fa$.pred)
mape(test_data_fa, Salary_Y, boost_preds_fa$.pred)

boost_pred_table_fa <- test_data_fa %>%
  mutate(Salary_Pred = boost_preds_fa$.pred,
         Per_error = (boost_preds_fa$.pred - Salary_Y) / Salary_Y,
         Salary_diff= boost_preds_fa$.pred - Salary_Y) %>%
  select(c(Player, MLS_C, Salary_Y, Salary_Pred, Salary_diff, Per_error))

combined_boost_output <- test_data %>%
  left_join(boost_pred_table_rook, by = 'Player', suffix = c(".x", "")) %>%
  select(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred) %>%
  left_join(boost_pred_table_fa, by = 'Player', suffix = c("", ".y")) %>%
  mutate(MLS_C = ifelse(is.na(MLS_C), MLS_C.y, MLS_C),
         Salary_C = ifelse(is.na(Salary_C), Salary_C.y, Salary_C),
         Salary_Y = ifelse(is.na(Salary_Y), Salary_Y.y, Salary_Y),
         Salary_Pred = ifelse(is.na(Salary_Pred), Salary_Pred.y, Salary_Pred)) %>%
  select(Player, MLS_C, Salary_C, Salary_Y, Salary_Pred)

save(list = c('combined_reg_output', 'combined_rf_output', 'combined_boost_output', 'final_reg_rook', 'final_reg_fa', 
              'final_rf_rook', 'final_rf_fa', 'final_boost_rook', 'final_boost_fa'), file = "data/02-batter-analysis.rda")
