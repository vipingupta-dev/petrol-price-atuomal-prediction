# Petrol Price AutoML Predictor
A Python project that predicts petrol prices using automated machine learning.
What makes it different from a simple price prediction model is that it automatically compares 20+ regression models using LazyPredict, tunes the best ones using GridSearchCV, and explains every prediction using SHAP values. I built this in Google Colab using real crude oil and exchange rate data pulled via yfinance.

---

## What it does
You give it current economic conditions — crude oil price, USD/INR exchange rate, and recent petrol price history. The pipeline engineers time series features, runs AutoML to find the best model, and predicts the petrol price per litre. It also tells you which economic factors drove that prediction and by how much.

---

## How it works
The pipeline runs through these steps:

1. **Load data** — pulls crude oil (Brent) and USD/INR data via yfinance, derives petrol price
2. **EDA** — plots price history, correlations, distributions, and missing value maps
3. **Clean** — forward fills missing values, caps outliers, extracts date features
4. **Feature engineering** — builds lag features, crude oil lag, price momentum, cyclical encodings
5. **Baseline models** — trains Naive and Random Forest baselines to set a performance floor
6. **AutoML** — runs LazyPredict to compare 20+ regressors, tunes top 3 with GridSearchCV
7. **Evaluate** — compares all models on RMSE, MAE, MAPE, R² with visualizations
8. **Explain** — computes SHAP values to show which features drive each prediction
9. **Save** — saves model, scaler, feature list, and metadata to checkpoints
10. **UI** — launches a Gradio web app where you can adjust economic conditions and get live predictions

---

## Results

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Naive (Yesterday's Price) | 2.53 | 1.99 | 2.80% | 0.23 |
| Random Forest (Manual) | 1.74 | 1.37 | 1.93% | 0.64 |
| **ExtraTreesRegressor (Tuned)** | **1.35** | **1.07** | **1.51%** | **0.78** |

AutoML winner improved RMSE by **46.6%** over the naive baseline.

---

## Top 5 Prediction Drivers (SHAP)

| Rank | Feature | Avg Impact |
|---|---|---|
| 1 | crude_oil_price | ₹8.86 per litre |
| 2 | year | ₹4.48 per litre |
| 3 | crude_lag_7 | ₹3.93 per litre |
| 4 | price_lag_7 | ₹2.94 per litre |
| 5 | price_lag_30 | ₹1.18 per litre |

Crude oil price is the single biggest driver — every $1 rise in crude adds ~₹8.86 to the predicted petrol price.

---

## Tech stack
- Python
- pandas, numpy
- scikit-learn
- LazyPredict
- SHAP
- matplotlib, seaborn
- yfinance
- Gradio
- Google Colab

---

## How to run it

**Option 1 — Google Colab (easiest)**
- Open `petrol_price_automl_predictor.py`
- Upload it to colab.research.google.com as a notebook or paste cells manually
- Run the install cell first, then restart runtime
- Run all remaining cells top to bottom
- The last cell launches a Gradio web app with a public link

**Option 2 — Run locally**
Make sure you have Python installed then run:
```
pip install -r requirements.txt
```
Then run the script:
```
python petrol_price_automl_predictor.py
```

---

## Requirements
All dependencies are listed in `requirements.txt`. Main ones are:
```
pandas
numpy
scikit-learn
lazypredict
shap
yfinance
matplotlib
seaborn
gradio
joblib
```

---

## Project structure
```
petrol-price-automl-predictor/
│
├── petrol_price_automl_predictor.py   # main script with all steps 0-10
├── requirements.txt                   # all dependencies
├── README.md                          # this file
└── .gitignore                         # files excluded from git
```

---

## What I learned building this

The biggest lesson was that lag features (yesterday's price, last week's price) are extremely powerful for price prediction but also dangerous — linear models can exploit them to get perfect R²=1 on smooth data, which looks great but is completely meaningless. I learned to filter out linear models from AutoML results when lag features are present and rely on tree-based models that capture the non-linear relationships between crude oil, exchange rates, and retail prices.

I also learned that time series data must never be split randomly. Using a random train/test split on price data leaks future information into training and produces unrealistically good metrics. Time-based splitting — training on older data and testing on the most recent data — gives honest results that reflect real-world performance.

---

## Author
Vipin Gupta
vipingupta.dev@gmail.com
github.com/vipingupta-dev
