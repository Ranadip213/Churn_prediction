# 📊 Telco Customer Churn Prediction (PySpark)

This project predicts **customer churn** (whether a telecom customer will leave the company) using **PySpark MLlib Logistic Regression**.  
The project uses a preprocessing + model pipeline, so you can feed raw Telco customer data and directly get churn predictions.

---

## 🚀 Features
- Data preprocessing with **StringIndexer** and **VectorAssembler**
- Logistic Regression classifier with Spark MLlib
- Pipeline training & saving (`PipelineModel`)
- Load saved pipeline and run predictions on raw CSV input
- Save predictions to CSV for downstream use

---


## 🛠 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Ranadip213/Telco_Churn_Prediction.git
   
Install required dependencies:
```
pip install pyspark pandas
```
Verify Spark installation:
```
pyspark --version
```
📊 Dataset
We use the Telco Customer Churn dataset, which contains customer information such as:
Demographics: gender, senior citizen, partner, dependents
Account info: tenure, contract type, payment method
Services: phone service, internet service, streaming services
Charges: monthly charges, total charges
Churn: Yes/No (target column)
▶️ Usage
1. Train Model
Run the training script to build the preprocessing + logistic regression pipeline and save it:
python train_churn_model.py
This saves the pipeline to:
/content/model/churn_pipeline
2. Run Predictions
Run inference on a raw Telco dataset:
```
python prediction.py
```
Loads pipeline from /content/model/churn_pipeline
Reads WA_Fn-UseC_-Telco-Customer-Churn.csv
```
Outputs predictions to predictions.csv
```
📈 Results
Example prediction output:
customerID	probability	prediction
7590-VHVEG	[0.82, 0.18]	0
5575-GNVDE	[0.23, 0.77]	1
probability: likelihood of no churn (0) vs churn (1)
prediction: 0 = no churn, 1 = churn

---

💡 Future Improvements
Hyperparameter tuning with ParamGridBuilder + CrossValidator
Try other classifiers (Random Forest, Gradient Boosted Trees)
Deploy pipeline as a REST API (Flask/FastAPI)
Add visualization dashboard with Streamlit
---
👨‍💻 Author
Ranadip Gope
Contact: ranadipgope157@example.com
