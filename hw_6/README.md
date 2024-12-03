
# Logistic Regression Model for Rain Prediction

This project focuses on building and evaluating a logistic regression model to predict the likelihood of rain in Australia for the next day.

## Dataset
The dataset used is the **Rain in Australia** dataset, which includes various meteorological features to predict rainfall:
- Numerical features such as temperature, humidity, wind speed, etc.
- Categorical features like wind direction, location, etc.

## Preprocessing Steps
1. **Missing Value Handling**:
   - Numerical features: Imputed using the mean.
   - Categorical features: Converted to appropriate string types and encoded.

2. **Feature Scaling**:
   - Numerical features were normalized using `StandardScaler`.

3. **Feature Encoding**:
   - Categorical features were one-hot encoded using `OneHotEncoder`.

4. **Class Balancing**:
   - Logistic regression was trained with `class_weight='balanced'` to address the class imbalance in the dataset.

## Model Evaluation
1. **Metrics**:
   - Precision, Recall, F1-score, and Accuracy were calculated.
   - The model achieved:
     - Accuracy: **85%** (without class balancing).
     - Improved Recall for the minority class with class balancing.

2. **ROC Curve**:
   - The ROC-AUC score of **0.86** demonstrates good discrimination capability.

## Key Observations
1. Balancing the dataset improved the Recall for the minority class but slightly reduced overall precision.
2. AUC of 0.86 indicates the model is effective at classifying rain predictions.

## Future Work
- Experiment with other models (e.g., Random Forest, Gradient Boosting).
- Optimize the classification threshold based on the ROC curve.
- Perform hyperparameter tuning for further improvement.

## Requirements
- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `sklearn`
  - `matplotlib`

## How to Run
1. Clone the repository.
2. Install the required libraries.
3. Run the notebook/script for preprocessing and training the model.

## Contributions
Feel free to fork the repository and contribute by adding more advanced models or improving preprocessing techniques.

## License
This project is open-source and available under the MIT License.
