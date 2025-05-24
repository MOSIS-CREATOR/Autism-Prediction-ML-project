# Autism Prediction Using Machine Learning

This project uses machine learning techniques to predict the likelihood of Autism Spectrum Disorder (ASD) based on questionnaire data and other input features.

## Project Overview

The notebook contains:

- Importing the Dependencies
For clarity and reproducibility, all necessary libraries (e.g., pandas, sklearn, xgboost, seaborn, etc.) are imported at the beginning.

- Data Loading and Understanding
The dataset is loaded and inspected:

Shapes, data types, and missing values.

Basic statistics using .describe() and .info().

- Exploratory Data Analysis (EDA)
Visual and statistical exploration to understand:

Class distributions.

Feature relationships.

Correlations (heatmaps, countplots, etc.).

Detection of outliers or imbalances.

- Data Preprocessing
Dropping irrelevant features (ID, age_desc, etc.).

Encoding categorical variables.

Handling missing values.

SMOTE to balance the dataset.

- Model Training
Multiple models trained for comparison, such as:

Logistic Regression

Decision Tree

XGBoost (etc.)

- Model Selection and Hyperparameter Tuning
Evaluating models with cross-validation.

Using techniques like RandomizedSearchCV or manual tuning to optimise model parameters.

- Model Evaluation
Accuracy, Precision, Recall, F1 Score.

Confusion matrix.

Discussion on performance and potential improvements.

## Dataset

This project uses a dataset from Kaggle.

You can download it from (You have to accept the late submission competition rule): https://www.kaggle.com/competitions/autismdiagnosis

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

## ML Models Used

- Decision Tree
- Random Forest
- XGBoost

## Results

Accuracy score:
 0.84375

Confusion matrix:
 [[113  15]
 [ 10  22]]

Classification report:
               precision    recall  f1-score   support

           0       0.92      0.88      0.90       128
           1       0.59      0.69      0.64        32

    accuracy                           0.84       160
   macro avg       0.76      0.79      0.77       160
weighted avg       0.85      0.84      0.85       160

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/mosis-creator/autism-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd autism-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the notebook:
   ```bash
   jupyter notebook Autism_Prediction_Using_Machine_Learning.ipynb
   ```

## License

This project is licensed under the GNU General Public License v3.0.
