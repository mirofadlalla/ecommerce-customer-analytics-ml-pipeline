# 💼 **E-Commerce Customer Analytics & Predictive Modeling**

### *An End-to-End Data Science Project by Omar Yaser*

📊 *From raw data → business insights → ML models → deployed APIs → Power BI executive dashboards*

---

## 🧩 **1. Project Overview**

This project represents a **complete real-world data science pipeline** — built from scratch to help an e-commerce company understand customer behavior, reduce product returns, and improve loyalty through predictive analytics and automation.

The analysis covers data from **2020 → 2025**, with **~400K orders, ~150K customers, and $49.8M total revenue**.
All insights, models, and dashboards were created independently using **Python (Pandas, Scikit-learn, XGBoost)** and **Power BI**.

---

## 🚨 **2. Business Problem**

The company faced multiple challenges that directly affected profitability and growth:

| Problem                                  | Description                                                                          | Impact                         |
| ---------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------ |
| 🧾t **High return & cancellation rates** | Over **66K returned items** and **71K cancellations** causing revenue loss > **$6M** | Revenue loss & inventory waste |
| 💸 **Customer retention issues**         | Only **5–7%** of new customers remain active after the first order                   | Lost potential lifetime value  |
| 🚚 **Delivery & product issues**         | Many returns due to *late delivery*, *wrong size*, or *“not as described”*           | Poor customer satisfaction     |
| 🎯 **No targeted marketing**             | No CLV segmentation or loyalty tracking                                              | Inefficient marketing spend    |
| 🤖 **Lack of automation**                | Manual analysis, no ML support for predictions                                       | Slow decision-making           |

---

## 🌟 **3. Business Goals**

The goal of the project was to transform raw operational data into actionable intelligence and deploy AI-powered services to assist decision-making.
Specifically:

1. 🧹 **Clean & prepare data** for modeling with intelligent imputation (e.g., gender prediction).
2. 📈 **Analyze customer & order behavior** to uncover insights (loyalty, retention, returns).
3. 🤖 **Build machine learning models** to predict:

   * Customer loyalty
   * Order return probability
   * Customer lifetime value (CLV)
   * Customer segmentation (via clustering)
4. 🌐 **Deploy predictive APIs** using **FastAPI** for real-time use.
5. 📊 **Visualize insights** in a Power BI dashboard for executives and stakeholders.

---

## 🧠 **4. Data Overview**

| Dataset     | Description                                        |
| ----------- | -------------------------------------------------- |
| `customers` | Demographics, gender, country, loyalty, engagement |
| `orders`    | Order status, dates, value, cancellations, returns |
| `products`  | Product category, brand, price                     |
| `returns`   | Return reasons, quantities, loss amounts           |
| `sessions`  | Customer browsing sessions, devices, duration      |
| `payments`  | Payment methods, success rates                     |

🗓 **Time period:** Aug 2020 → 2025
🧟 **Customers:** ~150K
📦 **Orders:** ~400K
💰 **Revenue:** ~$49.8M

---

## 🧹 **5. Data Cleaning & Preprocessing**

Cleaning wasn’t just about dropping NaNs — it used **machine learning-based imputation**, **regex validation**, and **advanced feature engineering** to ensure high-quality data.

### 🔧 Steps Taken:

* **ydata_profiling:** Generated full profiling report for initial data audit (missing %, skewness, correlation, dtypes).
* **Encoding fixes:** Standardized text casing, stripped spaces, fixed typos in categories & payment methods.
* **Regex validation:** Used Regular Expressions to validate and fix invalid email formats.
* **Date handling:** Converted timestamps to datetime, extracted `year`, `month`, `day`, and `week`.
* **Outlier handling:** Applied IQR method to cap outliers in **Quantity** and **Price** columns.
* **Duplicate removal:** Verified unique keys (`order_id`, `customer_id`).
* **Feature engineering:** Derived features like `cancel_rate`, `return_rate`, `avg_spend`, `engagement_score`, `Recency`.
* **Gender imputation with ML 🧠:**
  When `gender` was missing, a trained **classification model** predicted it using the customer’s name and behavioral metrics (order frequency, spend, engagement).

### ✅ Output:

* Clean, consistent dataset with **0% missing critical fields.**
* Data exported as `.pkl` for reuse across EDA and modeling.

---

## 🔍 **6. Exploratory Data Analysis (EDA)**

### 📊 General Insights:

* **Total revenue:** $49.78M
* **Total orders:** 400K
* **Total customers:** 150K (unique ~139K)
* **Average order value (AOV):** 110.8
* **Returned items:** 66,569
* **Returning customers:** ~111K (80%)

---

### 💅 Product & Category Insights:

| Category             | Revenue Contribution | Key Insight                                                    |
| -------------------- | -------------------- | -------------------------------------------------------------- |
| 💅 **Beauty**        | **41.5%**            | Top revenue driver but high returns (esp. allergic reactions). |
| 👗 **Fashion**       | **27%**              | Strong revenue + high returns due to sizing issues.            |
| 💻 **Electronics**   | **15%**              | Moderate returns; mainly defective/late deliveries.            |
| 🏠 **Home & Sports** | **10%**              | Low return rates, consistent sales.                            |

➡️ **Conclusion:** Beauty & Fashion dominate both sales and returns — high value but high risk.

---

### 🔄 Return Behavior Insights:

* **Top reasons for returns:**

  1. Found better price elsewhere
  2. Allergic reaction
  3. Not as described / wrong size
  4. Damaged item / defective
  5. Late delivery

* Most returns were for **low-priced items**, not expensive ones → root cause = **policy & quality**, not price.

* **Allergic reactions** concentrated in Beauty category → supplier verification needed.

* **Return rates per reason** roughly even (~7K each) → systemic issue, not isolated.

---

### 🕛 Retention & Customer Lifecycle:

* Majority of customers = one-time buyers.
* Only **5–7%** stay active.
* Loyal customers (3+ completed orders) = **53.9K**.
* Loyal **AOV = 113.2**, non-loyal **AOV = 92.3** → loyalty driven by frequency, not value.

**Statistical tests:**

* ✅ **T-test:** Loyal vs Non-Loyal AOV difference significant (p=0.0).
* ❌ **Chi-square:** Gender vs Return behavior not significant.
* ✅ **ANOVA:** Spending differs significantly by country.

---

### 🌍 Time Trends:

* Revenue spike mid-2021 (marketing campaign or expansion).
* Customer acquisition peaks early in year, declines later → retention issue.
* Seasonal return peaks in Q4 (holiday period).

---

## 🤖 **7. Machine Learning Modeling**

| Task                     | Model               | Purpose                             | Performance        |
| ------------------------ | ------------------- | ----------------------------------- | ------------------ |
| 🧟 Customer Loyalty      | Logistic Regression | Predict loyal customers (>2 orders) | F1=0.84            |
| 🔁 Order Return          | XGBoost Classifier  | Predict if order will be returned   | ROC-AUC=0.89       |
| 💰 CLV Prediction        | XGBoost Regressor   | Predict Customer Lifetime Value     | RMSE=41.2, R²=0.83 |
| 👥 Customer Segmentation | KMeans              | Cluster customers by RFM metrics    | 3 clusters         |

### 🔧 Modeling Highlights:

* **Pipelines** with preprocessing + SMOTE + scaling.
* **GridSearchCV / RandomizedSearchCV** for hyperparameter tuning.
* **Cross-validation** for stable metrics.
* **SHAP values** for interpretability.

---

## 👥 **8. KMeans Clustering Results**

### 👥 **Cluster Summary:**

| Cluster | CLV | Frequency | Recency | AOV | Return Rate | Description                     |
| ------- | --- | --------- | ------- | --- | ----------- | ------------------------------- |
| 0       | 474 | 4         | 266     | 113 | 1           | Active, stable spenders         |
| 1       | 150 | 2         | 685     | 74  | 2           | Low-value, disengaged customers |
| 2       | 581 | 2         | 697     | 231 | 1           | High-value, infrequent VIPs     |

### 🧠 **Cluster Recommendations:**

* 🔹 **Cluster 0:** Maintain loyalty, promote new products.
* 🔹 **Cluster 1:** Winback campaigns, discount reactivation.
* 🔹 **Cluster 2:** VIP engagement, exclusive offers.

### ⚙️ **Technical Highlights:**

* PCA used for feature reduction (2D visualization).
* Pipeline-based preprocessing for automation.
* SHAP analysis for model explainability.
* SMOTE balancing for classification.

---

## 🌐 **9. API Deployment (FastAPI)**

### 🚀 Endpoints:

| Endpoint                    | Description                          |
| --------------------------- | ------------------------------------ |
| `/predict-customer-loyalty` | Predict if a customer is loyal       |
| `/customer-order-return`    | Predict if an order will be returned |
| `/customer-clv`             | Predict customer lifetime value      |
| `/customer-segment`         | Assign segment via K-Means           |

### 🔐 Features:

* API key authentication (`X-API-Key`)
* Async prediction with `asyncio.to_thread`
* Rate limiting system
* `/health` endpoint for status check

✅ Tested successfully using Postman.

---

## 📊 **10. Power BI Executive Dashboard**

Interactive dashboards designed for business decision-makers.

### 🔸 Dashboards:

1. **Customer Dashboard:** Loyalty, CLV, segmentation, geography.
2. **Orders Dashboard:** Orders, revenue, seasonal trends.
3. **Product Dashboard:** Top categories & brands.
4. **Returns Dashboard:** Reasons, losses, frequency.

### 💥 Highlights:

* Dynamic slicers for country, gender, status.
* Treemaps for category contribution.
* CLV segmentation visuals (High/Mid/Low).
* Total return losses visualized (-6.4M).

---

## 🗳 **11. Key Insights Summary**

| Area                  | Insight                         | Action                          |
| --------------------- | ------------------------------- | ------------------------------- |
| 💅 Beauty Returns     | Highest allergic & defect rates | Audit suppliers                 |
| 💸 Price Sensitivity  | Not key reason for returns      | Add discounts, fix descriptions |
| 🚚 Delivery           | Late delivery frequent          | Improve logistics SLAs          |
| 🤝 Loyalty            | Only 5–7% retention             | Launch loyalty programs         |
| 🌍 Country Difference | Spending varies by country      | Tailored pricing per region     |

---

## 🧩 **12. Next Steps**

* 🧠 Deploy NLP model (Hugging Face / LLM) for customer review sentiment analysis.
* 🔄 Integrate FastAPI with frontend dashboards.
* 📈 Build forecasting models (Prophet / LSTM).
* 🗳 Add Model Cards & metadata tracking.
* 🎁 Expand Power BI dashboards with live data updates.

---

## 🏁 **13. Final Thoughts**

This project demonstrates **end-to-end data science expertise**:

> Data Cleaning → EDA → Modeling → Deployment → Visualization

It combines **technical depth** with **business storytelling**, turning raw data into real financial impact and actionable insights.

---

💬 *Created with ❤️ by **Omar Yaser** — Data Analyst & Aspiring ML Engineer*
📧 **LinkedIn:** Omar Fadlalla  |  💻 **Portfolio:** Coming soon
🚀 **#DataScience #MachineLearning #PowerBI #FastAPI**
