
# Boosted Regression Tree (BRT) modelling of fire severity
# Chapter 4 — Meteorological predictors
# 62 predictors, Gaussian response, 10-fold cross-validation

library(tidyverse)
library(gbm)
library(viridis)
library(dismo)
library(Metrics)

data_all <- read.csv(Sys.getenv("METEO_INPUT_CSV", "data/BRT_input.csv"))

head(data_all)
summary(data_all)

# BRT hyperparameter tuning — configuration tested
# Tree complexity, learning rate and bag fraction varied during tuning
set.seed(85786)

BRTree <- gbm.step(
  data = data_all,
  gbm.x = 1:62,
  gbm.y = 63,
  family = "gaussian",
  tree.complexity = 5,
  learning.rate = 0.004,
  bag.fraction = 0.75,
  max.trees = 10000,
  n.trees = 100,
  n.folds = 10
)

BRTree$cv.statistics
BRTree$fitted

# Model performance
resp <- data_all[[63]]
fit <- BRTree$fitted

modeldata <- data.frame(
  resp = resp,
  fit = fit,
  resid = BRTree$residuals
)

fit_lm <- lm(resp ~ fit, data = modeldata)
summary(fit_lm)

r2 <- summary(fit_lm)$r.squared
r2

ggplot(modeldata, aes(x = resp, y = fit)) +
  geom_smooth(method = "lm", colour = "red") +
  geom_point(alpha = 0.45, size = 2) +
  theme_bw() +
  coord_equal() +
  labs(x = "Observed severity", y = "Fitted severity") +
  annotate(
    geom = "text",
    x = Inf, y = Inf,
    label = paste0("~R^{2}==", round(r2, 2)),
    hjust = 1.70, vjust = 2,
    colour = "black", parse = TRUE, size = 6.5
  )

ggplot(modeldata, aes(x = fit, y = resid)) +
  geom_point(alpha = 0.45, size = 2) +
  theme_bw() +
  labs(x = "Fitted value", y = "Residual")

# Variable importance
varimp <- summary(BRTree)
varimp

ggplot(
  varimp,
  aes(y = rel.inf, x = reorder(var, rel.inf), fill = rel.inf)
) +
  geom_bar(stat = "identity") +
  coord_flip() +
  geom_text(
    aes(label = round(rel.inf, 2)),
    colour = "gray5",
    size = 2.5,
    position = position_nudge(y = 0.5)
  ) +
  scale_fill_viridis(option = "D", direction = -1) +
  labs(y = "Relative influence (%)", x = "Variables") +
  guides(fill = "none") +
  theme_classic()

write.csv(
  varimp,
  "feature_importance_BoostedRegressionTree_v2RS_comments.csv",
  row.names = FALSE
)

# Error metrics
round(rmse(resp, fit), 2)
round(mae(resp, fit), 2)
