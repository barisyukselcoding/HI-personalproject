# Credit Card Fraud Detection

This project implements a machine learning solution for detecting fraudulent credit card transactions. It uses a dataset of credit card transactions, applies data preprocessing techniques, and trains two different models (Logistic Regression and Random Forest) to classify transactions as fraudulent or non-fraudulent.

## Project Overview

Credit card fraud detection is a significant issue in the financial sector, with billions of dollars lost annually due to fraudulent transactions. This project aims to develop machine learning models to detect such fraudulent transactions in real-time, helping financial institutions mitigate losses and improve the security of their customers' financial data.

## Dataset

The dataset used in this project is the Credit Card Fraud Detection dataset, available on Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

**Note**: The current code reads the dataset from a local file. For production use, consider implementing secure API access to the data.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

You can install the required packages using pip:

```
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Project Structure

The project consists of several main steps:

1. Data Import and Exploration
2. Data Cleaning
3. Data Analysis
4. Data Visualization
5. Resampling (to address class imbalance)
6. Model Training and Evaluation

## Key Features

- Handles imbalanced dataset using RandomOverSampler
- Implements and compares two models: Logistic Regression and Random Forest
- Provides visualizations for better understanding of the data
- Uses cross-validation to ensure model robustness

## Data Analysis

The analysis revealed several important insights about the dataset:

1. **Class Imbalance**: Only 0.17% of the transactions in the dataset are fraudulent, indicating a highly imbalanced dataset.

2. **Transaction Amounts**:
   - Average fraudulent transaction amount: $122.21
   - Average non-fraudulent transaction amount: $88.29
   This suggests that fraudulent transactions tend to involve larger amounts.

3. **Time Distribution**: Visualization of transaction times showed that:
   - Non-fraudulent transactions have a bimodal distribution, suggesting two peak times for legitimate transactions.
   - Fraudulent transactions are more evenly distributed across time, not showing specific peak periods.

4. **Amount Distribution**: The distribution of transaction amounts for fraudulent transactions was visualized, helping to identify any patterns or unusual characteristics in fraudulent transaction amounts.

## Data Preprocessing

1. **Missing Values**: No missing values were found in the dataset.
2. **Duplicate Rows**: 1081 duplicate rows were identified. These were removed to prevent bias in the analysis.
3. **PCA Transformation**: Most features (V1-V28) were already transformed using PCA for confidentiality reasons.

## Handling Class Imbalance

To address the significant class imbalance, RandomOverSampler was used to oversample the minority class (fraudulent transactions) in the training data. This technique helped to balance the classes and improve model performance.

## Model Training and Evaluation

Two models were trained and evaluated:

1. Logistic Regression
2. Random Forest

**Evaluation Metrics**  
For imbalanced datasets like this, precision (proportion of correctly identified fraud cases) and recall (ability to detect fraud) are key metrics. These provide insights into how well the models capture fraudulent transactions while minimizing false positives.

Both models achieved perfect scores across all metrics (precision, recall, F1-score, and accuracy) on the test set. Cross-validation was also performed to ensure the robustness of these results.

**Results**  
Initial evaluation and cross-validation indicated perfect performance:

- Logistic Regression CV accuracy: 1.0000 (+/- 0.0000)
- Random Forest CV accuracy: 1.0000 (+/- 0.0000)

While these results are impressive, they may indicate a potential overfitting issue or an "easy to classify" dataset. Further testing on new, unseen data is recommended to ensure the models generalize well.

## Future Work

- **Secure API access**: Future iterations will include integrating API endpoints that follow security protocols such as OAuth or token-based authentication to protect sensitive transaction data during retrieval.
- **Testing on new, unseen data**: Models should be tested on different datasets to ensure robustness and avoid overfitting.
- **Other resampling techniques**: Explore alternative techniques like SMOTE or ADASYN to handle class imbalance more effectively.
- **User interface**: Develop a simple UI for real-time fraud detection based on the trained models.
- **Investigate model performance**: Given the perfect accuracy, investigate whether more complex feature engineering or model changes could provide deeper insights.

## Ethical Considerations and Limitations

- **Bias in Fraud Detection**: Machine learning models can sometimes unintentionally reflect biases present in the data. This could lead to unfair outcomes, such as incorrectly flagging legitimate transactions.
- **Privacy Concerns**: Handling sensitive transaction data comes with the responsibility of maintaining strong security practices, which is why implementing secure API access is a key focus for future work.
- **Limitations of Dataset**: The dataset’s PCA-transformed features and extreme class imbalance could limit the model’s generalizability. Testing on different datasets is necessary to ensure wider applicability.

## Contributors

[Baris Yuksel] - Data Analyst, Sustainability & Risk Specialist

Feel free to reach out if you want to contribute or discuss improvements to this project.

## Resources and Further Reading

- [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/)
- [imbalanced-learn: Handling Imbalanced Datasets](https://imbalanced-learn.org/stable/)
- [Pandas: Python Data Analysis Library](https://pandas.pydata.org/)
- [Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

