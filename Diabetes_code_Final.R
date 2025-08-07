# Final Year Project
# Predict diabetes risk using health behaviour data (BRFSS 2015)



#Installing and loading the packages 
install.packages("tidyverse")
install.packages("caret", dependencies = TRUE)
install.packages("randomForest")
install.packages("pROC")
install.packages("readr")
install.packages("vip")

library(readr)
library(vip)

# Load the dataset ---
health_data <- read.csv("diabetes_binary_health_indicators_BRFSS2015.csv")


#checking for the amount of people who have & do not have diabetes 
table(health_data$Diabetes_binary)


#Dropped variables that are not relevant for the model's outcome
library(tidyverse)
health_data <- health_data %>% 
  select(-Stroke, -HeartDiseaseorAttack, -MentHlth, -PhysHlth) %>%
  mutate( 
    #BMI categories
    BMI_groups = case_when(
      BMI < 18.5 ~ "Underweight",
      BMI < 25 ~ "Normal",
      BMI < 30 ~ "Overweight",
      TRUE ~ "Obese"
    ),
    BMI_groups = factor(BMI_groups, levels = c("Underweight", "Normal", "Overweight", "Obese")), 
    
    #Added age groups for better interpretability
    Age_group = case_when(
      Age <= 4 ~ "18-39",
      Age <= 8 ~ "40-59",
      TRUE ~ "60+"
    ), 
    Age_group = factor(Age_group, levels = c("18-39", "40-59", "60+")),
    
    #Risk scores
    Cardiovas_risk = HighBP + HighChol,
    Lifestyle_risk = Smoker + HvyAlcoholConsump + (1 - PhysActivity),
    
    #High- risk combinations
    High_risk = as.numeric((BMI >= 30) & (HighBP == 1) & (PhysActivity == 0))
    )


# Convert outcome to factor
health_data$Diabetes_binary <- factor(health_data$Diabetes_binary,
                                     levels = c(0,1),
                                     labels = c("No", "Yes"))

# Changing the binary variables to factors with labels
binary_vars <- c("HighBP", "HighChol", "CholCheck", "Smoker", "PhysActivity", 
                 "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", 
                 "NoDocbcCost", "DiffWalk", "Sex", "High_risk")

health_data[binary_vars] <- lapply(health_data[binary_vars], 
                                 function(x) factor(x, levels = c(0,1), labels = c("No", "Yes")))
                                   


#Checking the overall missing values in the columns
cat("Missing values check:\n")
na_check <- colSums(is.na(health_data))
print(na_check[na_check > 0]) #This shows columns with missing values

if(sum(na_check) == 0) cat("✓ No missing values found\n")

#downsizing my large dataset 
dataset_small <- health_data %>% sample_n(50000)

# Split the data into Train/Test Sets 
library(caret)
set.seed(123)
split_index <- createDataPartition(dataset_small$Diabetes_binary, p = 0.8, list = FALSE)
train_data <- dataset_small[split_index, ]
test_data <- dataset_small[-split_index, ]

# Setting up cross-validation
cross_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
)

#Training random forest model with cross-validation
rf_model_cross <- train(
  Diabetes_binary ~ .,
  data = train_data,
  method = "rf",
  trControl = cross_control,
  ntree = 100,
  importance = TRUE
)
 
#Showing the Cross-validation results
print(rf_model_cross)
cat("Cross-validation AUC:", round(max(rf_model_cross$results$ROC), 3), "\n")

#The first model is kept for compatibility
rf_model <- rf_model_cross$finalModel


# Trying to predict on Test Data
# Finding the probabilities and predictions with matching factor levels
testing_probs <- predict(rf_model_cross, newdata = test_data, type = "prob")[,"Yes"]
testing_pred <- ifelse(testing_probs > 0.5, "Yes", "No")
testing_pred <- factor(testing_pred, levels = c("No", "Yes"))  

# Evaluate the performance
library(pROC)
confusionMatrix(testing_pred, test_data$Diabetes_binary, positive = "Yes")
roc_obj <- roc(test_data$Diabetes_binary, testing_probs)
auc(roc_obj)

# --- Extract KPIs and Export to CSV ---

# Calculating confusion matrix and stats
conf_mat <- confusionMatrix(testing_pred, test_data$Diabetes_binary, positive = "Yes")

# Extract accuracy, precision, recall
accuracy <- conf_mat$overall["Accuracy"]
precision <- conf_mat$byClass["Precision"]
recall <- conf_mat$byClass["Recall"]

# Calculate AUC
roc_obj <- roc(test_data$Diabetes_binary, testing_probs)
auc_value <- auc(roc_obj)

# Create KPI data frame
kpi_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "AUC"),
  Value = c(as.numeric(accuracy), as.numeric(precision), as.numeric(recall), as.numeric(auc_value))
)

# Export KPIs to CSV
write.csv(kpi_df, "model_metrics.csv", row.names = FALSE)

cat("KPIs saved to model_metrics.csv\n")

plot(roc_obj, main = "ROC Curve - Random Forest")

#The importance of variable
library(randomForest)
varImpPlot(rf_model)

#Logistic regression as a benchmark
glm_model <- glm(Diabetes_binary ~ ., data = train_data, family = "binomial")
glm_probab <- predict(glm_model, newdata = test_data, type = "response")
glm_predi <- ifelse(glm_probab > 0.5, "Yes", "No")
glm_predi <- factor(glm_predi, levels = c("No", "Yes"))

confusionMatrix(glm_predi, test_data$Diabetes_binary, positive = "Yes")
roc_obj_glm <- roc(test_data$Diabetes_binary, glm_probab)
auc(roc_obj_glm)
plot(roc_obj_glm, main = "ROC Curve - Logistic Regression")

#Evaluating fairness by the subgroup (Gender)
test_data$Predicted_tag <- testing_pred

subgroup_sex <- test_data %>%
  group_by(Sex) %>%
  summarise(
    Accuracy = mean(Predicted_tag == Diabetes_binary),
    Sensitivity = sum(Predicted_tag == "1" & Diabetes_binary == "1") / sum(Diabetes_binary == "1"),
    Specificity = sum(Predicted_tag == "0" & Diabetes_binary == "0") / sum(Diabetes_binary == "0")
  )
print(subgroup_sex)

#Evaluating cost sensitive
confusion_mat <- confusionMatrix(testing_pred, test_data$Diabetes_binary, positive = "Yes")
FN <- confusion_mat$table["No", "Yes"]  # False Negatives: predicted No, actually Yes
FP <- confusion_mat$table["Yes", "No"]  # False Positives: predicted Yes, actually No
cost_FN <- 5
cost_FP <- 1
total_cost <- FN * cost_FN + FP * cost_FP
print(paste("Total cost of misclassification estimated:", total_cost))

#Important for feature visuals
library(vip)
vip(rf_model, num_features = 10, bar = TRUE)

# Checking the visual for class balance
ggplot(dataset_small, aes(x = Diabetes_binary, fill = Diabetes_binary)) +
  geom_bar() +
  labs(
    title = "Class Distribution: Diabetes Status",
    x = "Diabetes Status",
    y = "Count"
  ) +
  theme_minimal()

#Checking visuals for age group as it is a large risk factor
# If the age is continuous, it should binned 
dataset_small <- dataset_small %>%
  mutate(AgeGroup = cut(Age, breaks = c(17, 29, 39, 49, 59, 69, 79, 89), right = TRUE))

ggplot(dataset_small, aes(x = AgeGroup, fill = Diabetes_binary)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Diabetes Status by Age Group",
    x = "Age Group",
    y = "Proportion",
    fill = "Diabetes"
  ) +
  theme_minimal()

#Checking for visuals on behaviour risks
ggplot(dataset_small, aes(x = PhysActivity, fill = Diabetes_binary)) +
  geom_bar(position = "fill") +
  labs(
    title = "Diabetes Proportion by Physical Activity",
    x = "Physically Active",
    y = "Proportion",
    fill = "Diabetes Status"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal()

#A visual on healthcare access vs diabetes
ggplot(dataset_small, aes(x = AnyHealthcare, fill = Diabetes_binary)) +
  geom_bar(position = "fill") +
  labs(
    title = "Diabetes Proportion by Healthcare Access",
    x = "Has Healthcare Access",
    y = "Proportion",
    fill = "Diabetes Status"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal()

#A visual on heavy alcohol consumption vs diabetes
ggplot(dataset_small, aes(x = HvyAlcoholConsump, fill = Diabetes_binary)) +
  geom_bar(position = "fill") +
  labs(
    title = "Diabetes Proportion by Heavy Alcohol Use",
    x = "Heavy Alcohol Use",
    y = "Proportion",
    fill = "Diabetes Status"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal()


# Preparing for Tableau Export 
test_output <- test_data %>%
  mutate(
    Predicted_Prob = testing_probs,
    Predicted_tag = testing_pred,
    AgeGroup = cut(
      as.numeric(as.character(Age)),
      breaks = c(17,29,39,49,59,69,79,89),
      right = TRUE,
      include.lowest = TRUE,
      labels = c("18-29","30-39","40-49","50-59","60-69","70-79","80+")
    ),
    Diabetes_binary = factor(Diabetes_binary, levels = c("No", "Yes")),
    Predicted_tag = factor(Predicted_tag, levels = c("No", "Yes"))
  )


write_csv(test_output, "test_predictions_for_tableau.csv")
