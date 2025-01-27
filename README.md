
# Heart Disease Prediction System

## Project Overview
The **Heart Disease Prediction System** is a machine learning project that analyzes medical data to predict the likelihood of heart disease in individuals. It uses supervised learning techniques and multiple classifiers to achieve accurate predictions based on patient data.

## Dataset
The dataset contains 303 instances with 13 features and a target variable:
- **Features**: Age, Sex, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar, Resting ECG Results, Max Heart Rate, Exercise-Induced Angina, ST Depression, Slope of ST Segment, Major Vessels, Thalassemia.
- **Target**: Presence or absence of heart disease (1: Disease, 0: No Disease).

## Libraries Used
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn

## Methodology
1. **Data Preprocessing**:
   - Cleaned and standardized the dataset.
   - Explored correlations using a heatmap.
2. **Model Training**:
   - Split the data into training (80%) and test sets (20%).
   - Trained the following models:
     - Logistic Regression
     - Random Forest Classifier
     - Support Vector Machine (SVM)
3. **Feature Selection**:
   - Used Recursive Feature Elimination (RFE) with Random Forest to select top 10 features.
4. **Evaluation**:
   - Evaluated models using accuracy, confusion matrix, and classification reports.

## Results
- **Logistic Regression**: 88.52% accuracy
- **Random Forest**: 85.25% accuracy
- **SVM**: 70.49% accuracy

## Key Highlights
- Logistic Regression achieved the highest accuracy.
- Feature selection improved model interpretability without significant loss of accuracy.
- Visualized correlations and performance metrics for better insights.

## How to Run
1. Clone the repository.
2. Load the dataset (`heart_disease_data.csv`) in your working directory.
3. Run the Python script or Jupyter Notebook provided to preprocess the data and train models.
4. View the performance metrics and comparisons in the output.

## Future Work
- Experiment with additional machine learning models.
- Perform hyperparameter tuning for further optimization.
- Incorporate cross-validation for robust performance evaluation.

