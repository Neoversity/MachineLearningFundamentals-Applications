
# Logistic Regression Model on Rain in Australia Dataset

## Summary
This project involves building a logistic regression model to predict whether it will rain tomorrow in Australia using the "RainTomorrow" dataset. The key steps include data preprocessing, feature encoding, model training, and evaluation.

## Steps Performed
1. **Data Cleaning**:
   - Removed columns with excessive missing values.
   - Dropped duplicates in columns.
   - Target variable `RainTomorrow` was preprocessed.

2. **Feature Engineering**:
   - Split numerical and categorical features.
   - Handled missing values using `SimpleImputer`.
   - Encoded categorical features using `OneHotEncoder` with `drop='first'`.

3. **Model Training**:
   - Logistic regression model was trained using multiple solvers (`liblinear`, `lbfgs`, `newton-cg`).
   - The best solver achieved an accuracy of **85.09%**.

4. **Evaluation**:
   - Classification metrics and ROC-AUC were calculated.
   - AUC score of **0.87** indicates strong predictive performance.

## Results
- **Accuracy**: 85.09%
- **AUC**: 0.87

## Recommendations
- Analyze feature importance to understand key predictors.
- Experiment with other models (e.g., Random Forest, Gradient Boosting).
- Perform cross-validation for stability checks.

## Visualizations
- **ROC Curve**: Included in the analysis with AUC score.

## Instructions
- Clone the repository and run the provided Jupyter Notebook for detailed insights.
