# Install and load required packages
# install.packages('readxl')
# install.packages('ggplot2')
# install.packages('dplyr')
# install.packages('corrplot')
# install.packages('Amelia') # For missing data visualization
# install.packages('caret')
# install.packages('C50')
# install.packages('xgboost')
# install.packages('kernlab')
# install.packages('e1071') # Required by caret and kernlab
#install.packages('MLmetrics')
# install.packages('tidyr') # Needed for gathering data for ggplot2 boxplots
#install.packages('reshape2')

library(readxl)
library(ggplot2)
library(dplyr)
library(corrplot)
library(Amelia)
library(caret)
library(C50)
library(xgboost)
library(kernlab)
library(e1071)
library(MLmetrics) # This should work after installation
library(tidyr)
library(reshape2)

# --- 1. Load the Dry Beans Dataset from Excel ---
# You might need to adjust the file path
# setwd(dirname(file.choose())) # Use this line to interactively select the file the first time
# getwd() # Check the working directory if you used file.choose()

# Assuming the file is in your working directory or specify the full path
excel_file_path <- "Dry_Bean_Dataset.xlsx" # Change this if your file is elsewhere

# Read data from Excel file, assuming the first sheet
# read_excel automatically detects column types
data <- read_excel(excel_file_path, sheet = 1)

# --- 2. Data Exploration ---

cat("--- Data Exploration ---\n")

# Display the first few rows of the dataset
cat("\nHead of dataset:\n")
print(head(data))

# Display the structure of the dataset
cat("\nStructure of dataset:\n")
str(data)

# Check for missing data
cat("\nMissing values per column:\n")
print(colSums(is.na(data))) # More common way than apply/sum

# Visualize missing data (if any) - Requires graphical output device
missmap(data, col = c("red", "green"), legend =TRUE, main = "Missing Data Map")
# Note: The dataset description indicated no missing values, so this map should be all green.

# Summarize the numerical variables
cat("\nSummary of numerical variables:\n")
print(summary(data %>% select_if(is.numeric)))

# Create boxplots to visualize the distribution of numerical variables before scaling
# Excluding the Class column for this plot
cat("\nGenerating Boxplots (Before Scaling)...\n")
# Using ggplot2 for better aesthetics and handling compared to base R par/boxplot loop
# First, gather the numerical data (excluding Class) into long format
data_numeric <- data %>% select(-Class)
data_numeric_long <- data_numeric %>%
  tidyr::gather(key = "Feature", value = "Value") # Requires tidyr package (often comes with tidyverse/dplyr)

# Plot boxplots
ggplot(data_numeric_long, aes(x = Feature, y = Value)) +
  geom_boxplot() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplots of Numerical Features (Before Scaling)", y = "Value")
# Print the plot (or it shows automatically in interactive R/RStudio)
# print(ggplot_boxplot_before) # If you assign it to a variable

# Create a bar plot of the target variable
cat("\nGenerating Class Distribution Bar Plot...\n")
ggplot(data, aes(x = Class, fill = Class)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Distribution of Dry Bean Classes", y = "Frequency", x = "Class") +
  theme(legend.position = "none")
# print(ggplot_barplot) # If you assign it to a variable

# Boxplot of Area by Class
cat("\nGenerating Boxplot of Area by Class...\n")
ggplot(data, aes(x = Class, y = Area, fill = Class)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Distribution of Area by Dry Bean Class", y = "Area", x = "Class") +
  theme(legend.position = "none")
# print(ggplot_areaboxplot) # If you assign it to a variable

# Display class counts
cat("\nCounts per Class:\n")
print(table(data$Class))

# --- 3. Correlation Matrix ---

cat("\n--- Correlation Matrix ---\n")

# Select only numerical columns for correlation calculation
data_numerical_only <- data %>% select_if(is.numeric)

# Compute the correlation matrix
cor_matrix <- cor(data_numerical_only)

# Plot the correlation matrix using corrplot
cat("Generating Correlation Matrix Plot...\n")
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)

# Print the correlation matrix
cat("\nCorrelation Matrix values:\n")
print(round(cor_matrix, 2)) # Print rounded values for readability

# --- 4. Data Preprocessing ---

cat("\n--- Data Preprocessing ---\n")

# Separate features (X) and target (y) BEFORE scaling
X <- data %>% select(-Class)
y <- data$Class

# Convert the 'Class' column to a factor early - essential for classification
# read_excel might read it as character, so explicit conversion is good practice
y <- as.factor(y)
cat("\nTarget variable converted to factor.\n")
str(y)

# Apply Standardization (Z-score scaling) as implemented in your script
# This scales features to have a mean of 0 and standard deviation of 1
cat("\nApplying Standardization (Z-score scaling)...\n")
# Use preProcess from caret for consistent scaling on train/test splits later
preProc_scaler <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preProc_scaler, X)

# Check head and boxplots after scaling
cat("\nHead of scaled features:\n")
print(head(X_scaled))

cat("\nGenerating Boxplots (After Scaling)...\n")
X_scaled_long <- X_scaled %>%
  tidyr::gather(key = "Feature", value = "Value")

ggplot(X_scaled_long, aes(x = Feature, y = Value)) +
  geom_boxplot() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplots of Numerical Features (After Standardization)", y = "Scaled Value")
# print(ggplot_boxplot_after) # If you assign it to a variable

# Combine scaled features with the target variable
scaled_data <- cbind(X_scaled, Class = y)


# --- 5. Data Splitting (Stratified 80% train, 20% test) ---
# Use createDataPartition from caret for stratified splitting to maintain class proportions
cat("\nSplitting data into 80% training and 20% testing (stratified)...\n")
set.seed(123) # for reproducibility
trainIndex <- createDataPartition(scaled_data$Class, p = 0.8, list = FALSE)
train_data <- scaled_data[trainIndex, ]
test_data <- scaled_data[-trainIndex, ]

# Check proportions in split data
cat("\nProportion of classes in Training data:\n")
print(prop.table(table(train_data$Class)))
cat("\nProportion of classes in Test data:\n")
print(prop.table(table(test_data$Class))) # Should be similar to training data

# Separate features and target for train and test sets
train_X <- train_data %>% select(-Class)
train_y <- train_data$Class
test_X <- test_data %>% select(-Class)
test_y <- test_data$Class


# --- 6. Define Training Control for Hyperparameter Tuning ---
# Using 10-fold cross-validation
cat("\nSetting up 10-fold Cross-Validation for tuning...\n")
train_control <- trainControl(method = "cv", number = 10,
                              classProbs = TRUE, # Needed for some metrics like AUC if used, good practice
                              summaryFunction = multiClassSummary # Appropriate for multiclass accuracy, etc.
)

# --- 7. Train and Evaluate Models (with Hyperparameter Tuning) ---

# --- C5.0 Decision Tree ---
cat("\n--- Training C5.0 Model (with tuning) ---\n")
# Define parameter grid for C5.0 tuning.
# 'trials' (number of boosting iterations) and 'model' (tree or rules).
# Including winnow=FALSE as per your previous finding.
c50_grid <- expand.grid(trials = c(1, 10, 20, 30),
                        model = c("tree", "rules"),
                        winnow = FALSE) # Based on your previous result

# Hyperparameter tuning and training for C5.0
set.seed(123) # for reproducibility
c50_model <- train(train_X, train_y,
                   method = "C5.0",
                   tuneGrid = c50_grid,
                   trControl = train_control,
                   metric = "Accuracy") # Optimize for Accuracy

cat("\nBest C5.0 parameters found:\n")
print(c50_model$bestTune)
print(c50_model) # Uncomment to see detailed tuning results

# Predictions on the test set
cat("\nEvaluating C5.0 on Test Set...\n")
c50_predictions <- predict(c50_model, newdata = test_X)

# Confusion Matrix and Evaluation for C5.0
cat("Confusion Matrix for C5.0 (Test Set):\n")
c50_cm <- confusionMatrix(c50_predictions, test_y)
print(c50_cm)

# Store C5.0 Accuracy
c50_accuracy <- c50_cm$overall['Accuracy']
cat("C5.0 Test Accuracy:", c50_accuracy, "\n")


# --- XGBoost ---
cat("\n--- Training XGBoost Model (with tuning) ---\n")
# Define parameter grid for XGBoost tuning.
# Tuning a subset of common parameters. Be mindful this can take time.
xgb_grid <- expand.grid(nrounds = c(50, 100, 150), # Number of boosting rounds
                        max_depth = c(2, 3, 4),     # Maximum depth of trees
                        eta = c(0.1, 0.3),          # Learning rate
                        gamma = c(0, 0.1),          # Minimum loss reduction
                        colsample_bytree = c(0.7, 1), # Subsample ratio of columns per tree
                        min_child_weight = c(1, 3), # Minimum sum of instance weight
                        subsample = c(0.7, 1)       # Subsample ratio of the training instances
)

# Hyperparameter tuning and training for XGBoost
set.seed(123) # for reproducibility
# XGBoost in caret needs target variable as factor for classification
xgb_model <- train(train_X, train_y,
                   method = "xgbTree", # Using tree booster for classification
                   tuneGrid = xgb_grid,
                   trControl = train_control,
                   metric = "Accuracy") # Optimize for Accuracy

cat("\nBest XGBoost parameters found:\n")
print(xgb_model$bestTune)
# print(xgb_model) # Uncomment to see detailed tuning results

# Predictions on the test set
cat("\nEvaluating XGBoost on Test Set...\n")
xgb_predictions <- predict(xgb_model, newdata = test_X)

# Confusion Matrix and Evaluation for XGBoost
cat("Confusion Matrix for XGBoost (Test Set):\n")
xgb_cm <- confusionMatrix(xgb_predictions, test_y)
print(xgb_cm)

# Store XGBoost Accuracy
xgb_accuracy <- xgb_cm$overall['Accuracy']
cat("XGBoost Test Accuracy:", xgb_accuracy, "\n")


# --- Support Vector Machine (SVM) ---
cat("\n--- Training SVM Model (with tuning) ---\n")
# Define parameter grid for SVM tuning (Radial kernel).
# Tuning C (Cost) and sigma.
# These parameters are important for SVM performance.
svm_grid <- expand.grid(C = c(0.1, 1, 10, 100),
                        sigma = c(.001, .01, 0.1, 1)) # Sigma for Radial kernel

# Hyperparameter tuning and training for SVM
set.seed(123) # for reproducibility
# Using svmRadial method in caret for Radial Basis Function kernel
svm_model <- train(train_X, train_y,
                   method = "svmRadial",
                   tuneGrid = svm_grid,
                   trControl = train_control,
                   metric = "Accuracy") # Optimize for Accuracy

cat("\nBest SVM parameters found:\n")
print(svm_model$bestTune)
# print(svm_model) # Uncomment to see detailed tuning results

# Predictions on the test set
cat("\nEvaluating SVM on Test Set...\n")
svm_predictions <- predict(svm_model, newdata = test_X)

# Confusion Matrix and Evaluation for SVM
cat("Confusion Matrix for SVM (Test Set):\n")
svm_cm <- confusionMatrix(svm_predictions, test_y)
print(svm_cm)

# Store SVM Accuracy
svm_accuracy <- svm_cm$overall['Accuracy']
cat("SVM Test Accuracy:", svm_accuracy, "\n")


# --- 8. Compare Model Accuracies ---
cat("\n--- Model Comparison on Test Set Accuracy ---\n")

accuracy_comparison <- data.frame(
  Model = c("C5.0 Decision Tree", "XGBoost", "SVM Radial"),
  Accuracy = c(c50_accuracy, xgb_accuracy, svm_accuracy)
)

# Sort by accuracy
accuracy_comparison <- accuracy_comparison %>% arrange(desc(Accuracy))

print(accuracy_comparison)

# Identify the best model based on test accuracy
best_model_name <- accuracy_comparison$Model[1]
cat("\nModel with the highest accuracy on the test set:", best_model_name, "\n")

# --- 9. Display Class-wise Metrics (Sensitivity, Specificity, F1-score) ---
cat("\n--- Class-wise Performance Metrics ---\n")

# Extract and display metrics for C5.0
cat("\nC5.0 Class Metrics (Sensitivity, Specificity, F1-score):\n")
print(round(c50_cm$byClass[, c("Sensitivity", "Specificity", "F1")], 4))

# Extract and display metrics for XGBoost
cat("\nXGBoost Class Metrics (Sensitivity, Specificity, F1-score):\n")
print(round(xgb_cm$byClass[, c("Sensitivity", "Specificity", "F1")], 4))

# Extract and display metrics for SVM
cat("\nSVM Class Metrics (Sensitivity, Specificity, F1-score):\n")
print(round(svm_cm$byClass[, c("Sensitivity", "Specificity", "F1")], 4))


# --- 10. Plot Confusion Matrix Comparison ---
cat("\n--- Generating Confusion Matrix Comparison Plots ---\n")

# Function to prepare data for plotting a confusion matrix
prep_conf_matrix_for_plot <- function(cm, model_name) {
  cm_table <- as.data.frame(cm$table)
  # Rename columns for clarity and ggplot2
  colnames(cm_table) <- c("Prediction", "Reference", "Count")
  cm_table$Model <- model_name
  return(cm_table)
}

# Prepare data for each model
c50_cm_plot_data <- prep_conf_matrix_for_plot(c50_cm, "C5.0")
xgb_cm_plot_data <- prep_conf_matrix_for_plot(xgb_cm, "XGBoost")
svm_cm_plot_data <- prep_conf_matrix_for_plot(svm_cm, "SVM Radial")

# Combine data for plotting
all_cm_plot_data <- bind_rows(c50_cm_plot_data, xgb_cm_plot_data, svm_cm_plot_data)

# Create the confusion matrix plot comparison
ggplot(data = all_cm_plot_data, aes(x = Reference, y = Prediction, fill = Count)) +
  geom_tile() + # Create the colored tiles
  geom_text(aes(label = Count), vjust = 1) + # Add text labels for counts
  facet_wrap(~ Model, ncol = 3) + # Separate plots by Model
  scale_fill_gradient(low = "white", high = "steelblue") + # Color scale
  theme_minimal() +
  labs(title = "Confusion Matrix Comparison Across Models (Test Set)",
       x = "Actual Class", y = "Predicted Class") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(angle = 0)) +
  coord_equal() # Make tiles square

# --- 11. Plot F1-score Comparison ---
cat("\n--- Generating F1-score Comparison Plot ---\n")

# Extract F1-scores
c50_f1 <- as.data.frame(c50_cm$byClass[, "F1"]) %>%
  rename(F1 = `c50_cm$byClass[, "F1"]`) %>%
  mutate(Class = rownames(.), Model = "C5.0")

xgb_f1 <- as.data.frame(xgb_cm$byClass[, "F1"]) %>%
  rename(F1 = `xgb_cm$byClass[, "F1"]`) %>%
  mutate(Class = rownames(.), Model = "XGBoost")

svm_f1 <- as.data.frame(svm_cm$byClass[, "F1"]) %>%
  rename(F1 = `svm_cm$byClass[, "F1"]`) %>%
  mutate(Class = rownames(.), Model = "SVM Radial")

# Combine F1-scores
all_f1_scores <- bind_rows(c50_f1, xgb_f1, svm_f1)

# Create the F1-score comparison plot
ggplot(all_f1_scores, aes(x = Class, y = F1, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) + # Create grouped bar plot
  theme_minimal() +
  labs(title = "F1-score Comparison by Class and Model (Test Set)",
       x = "Dry Bean Class", y = "F1-score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# --- 12. Display Overall Confusion Matrix Statistics Comparison ---
cat("\n--- Overall Confusion Matrix Statistics Comparison ---\n")

# Extract overall statistics
c50_overall <- as.data.frame(t(c50_cm$overall)) %>% mutate(Model = "C5.0")
xgb_overall <- as.data.frame(t(xgb_cm$overall)) %>% mutate(Model = "XGBoost")
svm_overall <- as.data.frame(t(svm_cm$overall)) %>% mutate(Model = "SVM Radial")
# --- End of Script ---