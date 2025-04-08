# Fraud Detection Project

## Dataset
This project uses a dataset from Kaggle's **IEEE-CIS Fraud Detection** competition.  
Dataset link: [IEEE Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

From this dataset, the file **train_transaction.csv** was used, containing over **500,000 rows** of transaction data.

## Dependencies
The required dependencies for running this project are listed in `requirements.txt`.

## Usage
1. Ensure you have the necessary Python environment set up.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the fraud detection script:
   ```bash
   python Fraud_Detection.py
   ```

## Model & Approach
- Data preprocessing includes feature selection, encoding categorical variables, and normalizing numerical features.
- Feature engineering is applied to extract meaningful insights from transaction data.
- A combination of **XGBoost** and **Neural Networks (PyTorch)** is used for classification.
- SMOTE is applied to handle class imbalance.

## Results
- The model is trained using **95% of the dataset** and tested on **5%**.
- Performance metrics such as **accuracy, precision, recall, F1-score, and AUC-ROC** are computed.

## References
- **Kaggle Competition**: [IEEE Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

