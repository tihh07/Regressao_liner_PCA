# Carregando as bibliotecas necessárias
library(tidyverse)
library(tidymodels)
library(janitor)
library(caret)
library(MASS)
library(corrplot)

# ------ Carregamento e Preparação de Dados ------
# Carregando os dados
data(solubility, package = "AppliedPredictiveModeling")

# Limpeza e formatação dos dados
train_sol <- solTrainXtrans %>% mutate(solubility = solTrainY) %>% clean_names()
test_sol <- solTestXtrans %>% clean_names()
resposta <- solTestY

# ------ Análise Exploratória ------
# Estatísticas descritivas
train_sol %>% skimr::skim()

# ------ Modelagem Linear Simples ------
# Modelo base
mdl_lm_base <- lm(solubility ~ ., data = train_sol)
summary(mdl_lm_base)
plot(mdl_lm_base)

# Avaliação do Modelo Base
pred_lm_base <- predict(mdl_lm_base, newdata = test_sol)
compara <- data.frame(obs = resposta, pred = pred_lm_base)
caret::defaultSummary(compara)

# ------ Modelo Linear com Caret ------
# Treinamento e avaliação
mdl_lm_caret <- train(solubility ~ ., data = train_sol,
                      method = "lm",
                      trControl = trainControl(method = "cv", number = 5))

glance(mdl_lm_caret$finalModel)

# ------ Modelo Linear com PCA ------
# Treinamento e avaliação
mdl_lm_pca <- train(solubility ~ ., data = train_sol,
                    method = "lm",
                    preProcess = c("center", "scale", "pca"),
                    trControl = trainControl(method = "cv", number = 5))

glance(mdl_lm_pca$finalModel)

# ------ Modelo Linear com Correlação ------
# Treinamento e avaliação
mdl_lm_corr <- train(solubility ~ ., data = train_sol,
                     method = "lm",
                     preProcess = c("center", "scale", "corr"),
                     trControl = trainControl(method = "cv", number = 5))

glance(mdl_lm_corr$finalModel)

# ------ Redução de Dimensão com Tidymodels ------
# Criação da receita
rec_correlation <- recipe(solubility ~ ., data = train_sol) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_corr(all_numeric_predictors()) %>% 
  prep()

# Modelo Linear com Tidymodels
mdl_spec_tidym <- linear_reg()
wkfl_lm_tidym <- workflow() %>% 
  add_recipe(rec_correlation) %>% 
  add_model(mdl_spec_tidym)

# Treinamento e avaliação
mdl_fit_tidym <- fit(wkfl_lm_tidym, data = train_sol)
glance(mdl_fit_tidym)

# Reamostragem e ajuste do modelo
resample_tidym <- vfold_cv(train_sol, v = 5)
mdl_fit_resample_tidym <- fit_resamples(object = wkfl_lm_tidym,
                                        resamples = resample_tidym,
                                        control = control_resamples(save_pred = TRUE))

# Métricas finais
collect_metrics(mdl_fit_resample_tidym)
show_best(mdl_fit_resample_tidym, metric = "rmse")

