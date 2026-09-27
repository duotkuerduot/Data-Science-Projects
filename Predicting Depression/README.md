 # Predicting Depression (Mental Health)

This project focuses on analyzing and predicting depression using a dataset containing various mental health-related features. The work involves data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model development to classify individuals as depressed or not.

## Key Steps and Highlights

### 1. **Data Loading and Preprocessing**
- The dataset was loaded from a CSV file (`train.csv`).
- Renamed long column names for better readability (e.g., `Family History of Mental Illness` → `Family History`).
- Handled missing values by filling them with appropriate statistics (e.g., median for numerical columns).
- Converted categorical variables (e.g., `Gender`, `Dietary Habits`, `Suicidal Thoughts`) into numerical values for analysis.

### 2. **Exploratory Data Analysis (EDA)**
- **Correlation Matrix**: A heatmap was plotted to identify features most correlated with depression.
- **Visualizations**:
  - Pie charts were created to analyze the relationship between depression and features like suicidal thoughts, family history, CGPA, and pressure.
  - Bar plots were used to explore the distribution of depression across genders.

### 3. **Feature Engineering**
- Combined related columns (e.g., `Study Satisfaction` and `Job Satisfaction` into `Satisfaction`).
- Mapped professions and degrees to numerical levels for better analysis.

## Visualizations
- **Correlation Matrix**: Showed the relationships between features and depression.
- **Pie Charts**: Highlighted the distribution of depression across key features like suicidal thoughts, family history, CGPA, and pressure.
- **Bar Plots**: Explored gender-based depression distribution.

### 4. **Modeling**
- Defined features (`X`) and target (`y`) for classification.
- Standardized the data using `StandardScaler` to improve model performance.
- Split the data into training and testing sets using an 80-20 split.

### 5. **Machine Learning Models**
- Implemented and evaluated multiple models:
  - Logistic Regression
  - Decision Tree
  - XGBoost
  - Random Forest
- Metrics such as accuracy and ROC AUC scores were used to evaluate model performance.
- Visualized model performance using bar charts of ROC AUC scores.

### 6. **Hyperparameter Tuning**
- Used `GridSearchCV` to optimize hyperparameters for Logistic Regression.
- Performed cross-validation to validate model performance.

### 7. **Test Dataset**
- Preprocessed the test dataset (`test.csv`) similarly to the training data.
- Made predictions using the best-performing model and saved the results to submission.csv.

## Results
- The models achieved competitive accuracy and ROC AUC scores, with XGBoost and Random Forest performing particularly well.
- The final predictions were saved in a format suitable for submission.

## How to Use
1. Clone the repository and install the required libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`).
2. Run the solution.ipynb notebook to reproduce the analysis and predictions.
3. Modify the code to experiment with additional features or models.

This project provides a comprehensive approach to analyzing mental health data and predicting depression, with a focus on interpretability and actionable insights.
