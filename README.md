# Data Science Project - IST 2023/24

## Project Overview
The project consists of applying machine learning techniques to two distinct datasets, focusing on classification and forecasting tasks. Students explore different data preparation strategies, model architectures, and evaluation metrics to understand the impact of data-driven decisions in machine learning.

## Tasks
### 1. Classification
- **Datasets:**
  - **Health domain:** Classification of post-COVID patient data.
  - **Services domain:** Classification of credit scores.
- **Data Processing:**
  - Data profiling (dimensionality, distribution, sparsity, granularity).
  - Handling missing values, outliers, and scaling.
  - Feature selection and transformation techniques.
  - Model training using **Naïve Bayes, k-NN, Decision Trees, Random Forests, Multi-Layer Perceptrons, and Gradient Boosting**.
- **Evaluation:**
  - Model performance metrics.
  - Overfitting analysis and parameter tuning.

### 2. Forecasting
- **Datasets:**
  - **Health domain:** Forecasting COVID-19 death rates in Europe.
  - **Services domain:** Predicting traffic volume trends.
- **Data Processing:**
  - Time series aggregation, smoothing, differentiation.
  - Feature engineering and transformations.
- **Model Training:**
  - **Simple Average, Persistence, Rolling Mean, ARIMA, LSTMs**.
- **Evaluation:**
  - Performance metrics (MSE, RMSE, MAE, R2).
  - Comparative analysis of model performance.

## Installation & Setup
To set up the environment and run the models, follow these steps:

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/data-science-project.git
   cd data-science-project
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Classification Models**  
   ```bash
   python classification.py
   ```

4. **Run Forecasting Models**  
   ```bash
   python forecasting.py
   ```

## Results Summary
- **Best Classification Models:**
  - **Health Dataset:** Gaussian Naïve Bayes.
  - **Services Dataset:** k-NN with Manhattan distance (k=3).
- **Best Forecasting Models:**
  - **Health Dataset:** LSTMs, effectively capturing long-term trends.
  - **Services Dataset:** ARIMA, modeling cyclic patterns in traffic data.

## Critical Analysis
- Dataset characteristics significantly influence model performance (e.g., health vs. financial data).
- Data preprocessing choices impact final results (e.g., balancing, feature selection, and encoding methods).
- Overfitting tendencies were observed in some models, requiring careful regularization strategies.
- LSTMs provided superior results in forecasting, while ARIMA was a strong alternative for capturing cyclical patterns.

## License
This project is for educational purposes and follows an open-access policy.

## Contributors
- **Gonçalo Gonçalves**
- **José Cruz**
- **Jorge Santos**
- **Matilde Heitor**

