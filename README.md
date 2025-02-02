# Heart Disease Prediction using Machine Learning

## Overview
This project aims to develop a predictive model for heart disease using machine learning techniques. The dataset used for this analysis contains patient health records with various medical attributes. The goal is to analyze the dataset, perform exploratory data analysis (EDA), and build a model to classify patients as having heart disease or not.

## Dataset
This dataset consists of 11 features and a target variable. 

**1. Age:**  Patients Age in years <br>
**2. Sex:**  Gender of patient ; Male - 1, Female - 0<br>
**3. Chest Pain Type:**  Type of chest pain experienced by patient categorized into 1 typical, 2 typical angina, 3 non-anginal pain, 4 asymptomatic <br>
**4. resting bp s:**  Level of blood pressure at resting mode in mm/HG <br>
**5. cholestrol:**  Serum cholestrol in mg/dl <br>
**6. fasting blood sugar:**  Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false <br>
**7. resting ecg:**  Result of electrocardiogram while at rest are represented in 3 distinct values; 0 : Normal 1: Abnormality in ST-T wave 2: Left ventricular hypertrophy <br>
**8. max heart rate:**  Maximum heart rate achieved <br>
**9. exercise angina:**  Angina induced by exercise; 0 depicting NO 1 depicting Yes <br>
**10. oldpeak:**  Exercise induced ST-depression in comparison with the state of rest <br>
**11. ST slope:**  ST segment measured in terms of slope during peak exercise; 0: Normal 1: Upsloping 2: Flat 3: Downsloping <br>

#### Target variable
**12. target:** The target variable to predict; 1 means patient is suffering from heart risk and 0 means patient is normal.

## Exploratory Data Analysis (EDA)
The EDA process includes:

- Data cleaning and Preprocessing.
- Statistical analysis of feature distributions.
- Handling outliers.
- Correlation analysis between features.

## Machine Learning Models
The following machine learning models were implemented:

- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- MLPClassifier
- XGBoost Classifier

## Model Evaluation
Each model was evaluated using the following metrics:

- Accuracy
- Precision
- Sensitivity
- Specificity
- F1 Score
- ROC
- Log_Loss
- Mathew_Corrcoef

## Tools & Technologies Used
- Python
- Pandas, NumPy (Data Manipulation)
- Matplotlib, Seaborn (Visualization)
- Scikit-learn, XGBoost (Machine Learning)
- Jupyter Notebook
