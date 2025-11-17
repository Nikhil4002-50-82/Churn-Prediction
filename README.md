# Churn Prediction Project

This project focuses on building a deep learning model to predict customer churn using the **Telco Customer Churn** dataset. The workflow includes data acquisition, preprocessing using pipelines and transformers, exploratory profiling, model training with a neural network, and final evaluation of predictions.

## Dataset

The dataset used in this project is publicly available on Kaggle:

**Telco Customer Churn Dataset**
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Workflow

### 1. Kaggle API Setup

The workflow begins by uploading the `kaggle.json` file into Google Colab to authenticate access to Kaggle datasets. The file is placed in the appropriate configuration directory, and permissions are set so the Kaggle API can be used without issues.

### 2. Dataset Download

Using the Kaggle API, the Telco Customer Churn dataset was downloaded. The dataset was provided as a zip archive, which was extracted to obtain the CSV file needed for analysis.

### 3. Raw Data Profiling

YData Profiling was installed and used to generate an initial profiling report. This step provided insights into data quality issues, distributions, correlations, and the overall structure of the dataset.

### 4. Feature and Target Separation

The dataset was split into:

* **X**: All feature columns
* **y**: The churn column (target variable)

### 5. Data Type Fixing and Cleaning

The `TotalCharges` column, originally stored as an object, was converted to a float type. Empty strings in this column were replaced with `np.nan` so they could be properly handled as missing values later in the pipeline.

### 6. Train–Test Split

The complete dataset was divided into training and testing sets using scikit-learn's train-test split functionality.

### 7. Encoding the Target Variable

The churn column in **y** was encoded using LabelEncoder, converting the categorical churn labels into numerical values required for model training.

### 8. Preprocessing Pipelines

Two separate preprocessing pipelines were created:

* **Categorical Pipeline**

  * OneHotEncoder with the first category dropped
* **Numerical Pipeline**

  * SimpleImputer using the mean strategy (to handle missing values, particularly in `TotalCharges`)
  * StandardScaler applied to numerical features

### 9. ColumnTransformer Setup

A ColumnTransformer was used to combine the pipelines, applying:

* The categorical pipeline to categorical columns
* The numerical pipeline to numerical columns

### 10. Final Preprocessing Pipeline

A final pipeline was built that included the ColumnTransformer.
This pipeline was fitted on the training data and used to transform both training and testing sets, ensuring consistent preprocessing.

### 11. Neural Network Model Training

A Keras Sequential model was developed with the following architecture:

* Input layer: 64 units
* Hidden layers: 32 units → 16 units
* Activation function for hidden layers: ReLU
* Output layer: 1 unit with sigmoid activation (binary classification)

The model was trained on the preprocessed training data and produced an accuracy of approximately **83%**.

### 12. Post-Processing and Reporting

A profiling report was generated for the fully preprocessed dataset to inspect how the data appeared after all transformations.

### 13. Final Prediction and Output

The trained model produced predictions for the test set.
The predicted values were combined with the actual churn values (from y_test) into a single DataFrame, allowing a clear comparison of expected vs. predicted outcomes.
