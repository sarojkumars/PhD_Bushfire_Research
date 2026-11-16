
# Chapter 4 — Meteorological variable selection
# Fuzzy Forest with WGCNA and independent test evaluation

library(fuzzyforest)
library(randomForest)
library(WGCNA)
library(caTools)

set.seed(32)

# Load data
fires <- read.csv(Sys.getenv("METEO_INPUT_CSV", "data/FuzzyForest_input.csv"))

response_col <- "FireSev2"
predictor_cols <- setdiff(names(fires), response_col)

fires[[response_col]] <- as.factor(fires[[response_col]])

# Training and test split
split <- sample.split(fires[[response_col]], SplitRatio = 0.8)

train <- fires[split, , drop = FALSE]
test <- fires[!split, , drop = FALSE]

X_train <- train[, predictor_cols, drop = FALSE]
y_train <- train[[response_col]]

X_test <- test[, predictor_cols, drop = FALSE]
y_test <- test[[response_col]]

# Fuzzy Forest parameter configuration
WGCNA_params <- WGCNA_control(
  power = 6,
  TOMType = "unsigned",
  minModuleSize = 10,
  numericLabels = TRUE,
  pamRespectsDendro = FALSE
)

mtry_factor <- 1
drop_fraction <- 0.25
number_selected <- 20
keep_fraction <- 0.25
min_ntree <- 5000
ntree_factor <- 5
final_ntree <- 5000

screen_params <- screen_control(
  drop_fraction = drop_fraction,
  keep_fraction = keep_fraction,
  min_ntree = min_ntree,
  mtry_factor = mtry_factor,
  ntree_factor = ntree_factor
)

select_params <- select_control(
  drop_fraction = drop_fraction,
  number_selected = number_selected,
  min_ntree = min_ntree,
  mtry_factor = mtry_factor,
  ntree_factor = ntree_factor
)

# Fuzzy Forest model
wff_fit <- wff(
  X_train,
  y_train,
  WGCNA_params = WGCNA_params,
  screen_params = screen_params,
  select_params = select_params,
  final_ntree = final_ntree,
  num_processors = 4
)

print(wff_fit)
modplot(wff_fit)

saveRDS(wff_fit, "FuzzyForest_model.rds")

# Training predictions
pred_train <- predict(
  wff_fit,
  new_data = X_train,
  type = "prob"
)

predTrain <- cbind(
  train,
  as.data.frame(pred_train)
)

# Independent test predictions
pred_test <- predict(
  wff_fit,
  new_data = X_test,
  type = "prob"
)

predTest <- cbind(
  test,
  as.data.frame(pred_test)
)

# Export predictions
write.csv(
  predTrain,
  "prediction_FuzzyForest_train.csv",
  row.names = FALSE
)

write.csv(
  predTest,
  "prediction_FuzzyForest_test.csv",
  row.names = FALSE
)

# Test classification accuracy
pred_class <- predict(
  wff_fit,
  new_data = X_test,
  type = "response"
)

confusion_matrix <- table(
  Observed = y_test,
  Predicted = pred_class
)

print(confusion_matrix)

test_accuracy <- mean(as.character(pred_class) == as.character(y_test))

cat("Test accuracy:", round(test_accuracy, 4), "\n")
