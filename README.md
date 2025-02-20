# E-Commerce Churn User Prediction and Segmentation_Python

## 📌 Table of Contents

- [Background & Overview](#background--overview)
- [Dataset Description & Data Structure](#dataset-description--data-structure)
- [Design Thinking Process](#design-thinking-process)
- [Key Insights & Visualizations](#key-insights--visualizations)
- [Final Conclusion & Recommendations](#final-conclusion--recommendations)

---

## 📌 Background & Overview

### 🎯 Objective
Customer churn is a critical issue for businesses, as losing customers leads to revenue loss and increased acquisition costs. This project analyzes churn patterns using machine learning and provides strategic recommendations to reduce churn. Identifying key differences helps the company design personalized promotions and improve customer retention.

### 👤 Target Audience
- 📊 Marketing Analysts & Data Analysts
- 🏢 Business Managers & Decision-Makers
- 🤝 Customer Retention Teams

### ❓ Business Questions
- What factors contribute to customer churn?
- Can we predict which customers are likely to leave?
- How can we segment churned customers for targeted retention efforts?

### 🎯 Project Outcome
- Identified key factors influencing churn, such as engagement and order history.
- Built a predictive model to classify customers as likely to churn or not.
- Segmented churned customers to tailor retention strategies.

---

## 📂 Dataset Description & Data Structure

### 📌 Data Source
- **Source:** Synthetic dataset representing an e-commerce business.
- **Size:** 5,700 rows, 18 columns.
- **Format:** `.csv`

### 📊 Data Structure & Relationships

| Column Name            | Data Type  | Description                                      |
|------------------------|-----------|--------------------------------------------------|
| `CustomerID`          | INT       | Unique identifier for each customer             |
| `Churn`               | BOOL      | Whether the customer has churned (1 = Yes, 0 = No) |
| `Tenure`              | INT       | Number of months the customer has been active  |
| `PreferredLoginDevice`| TEXT      | Device used for login                          |
| `CityTier`            | INT       | Classification of customer’s city (1,2,3)       |
| `WarehouseToHome`     | FLOAT     | Distance from warehouse to home (km)           |
| `PreferredPaymentMode`| TEXT      | Payment method preferred by the customer       |
| `HourSpendOnApp`      | FLOAT     | Average hours spent on the app                 |
| `OrderCount`         | INT       | Total number of orders placed                  |
| `CashbackAmount`     | FLOAT     | Total cashback received                        |

---

## 🧠 Design Thinking Process

### 1️⃣ Empathize
Understanding business concerns:
- Who is leaving?
- What are the reasons?
- What are their behaviors?

### 2️⃣ Define Point of View
- Analyzing past customer behavior to predict churn probability.
- Identifying behavioral patterns to recommend retention strategies.
- Segmenting customers based on historical data.

### 3️⃣ Ideate
- Exploring different machine learning models and segmentation techniques to address churn.
- Comparing supervised and unsupervised learning approaches.

### 4️⃣ Prototype and Review
- Building and testing models using historical data.
- Evaluating models for accuracy and interpretability.

---

## ⚒️ Main Process

### 1️⃣ Import Necessary Libraries

- **Data Handling:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Preprocessing:** LabelEncoder, StandardScaler  
- **Machine Learning:** RandomForestClassifier, KMeans, AgglomerativeClustering  
- **Evaluation:** metrics  
- **Dimensionality Reduction:** PCA  

These libraries facilitate the cleaning, transforming, modeling, and evaluation of the dataset.

### 2️⃣ Exploratory Data Analysis (EDA)

#### Handling Data Issues:
- Identified and addressed missing or null values.
- Removed duplicate records.
- Ensured correct data types for analysis.

#### Feature Analysis:
- Univariate analysis for distributions of numerical and categorical variables.
- Bivariate analysis to determine correlations with churn.

#### Key Findings:
- Customers with fewer orders and lower engagement had a higher churn rate.
- Longer distance from warehouse to home increased churn probability.
- Payment preferences influenced churn behavior.

### 3️⃣ Supervised Model Training: Random Forest

#### Preprocessing Steps:
- Categorical variables encoded using Label Encoding.
- Applied SMOTE to balance class distribution.

#### Model Performance:
- Accuracy: **98%**
- Feature importance: **Tenure, Order Count, Cashback, and Warehouse Distance** were top predictors.

### 4️⃣ Unsupervised Learning: K-Means Clustering

#### Segmentation Strategy:
- Applied PCA for dimensionality reduction.
- Determined 4 optimal clusters using the elbow method.
- Used t-SNE for visualization.

---

## 📊 Key Insights & Visualizations

### 🔍 Key Findings
1️⃣ **Churn Drivers**
   - Customers with fewer orders and lower engagement had a higher churn rate.
   - Longer distance from warehouse to home increased churn probability.
   - Payment preferences influenced churn behavior.

2️⃣ **Prediction Performance**
   - Random Forest achieved **98% accuracy**, outperforming other models.
   - Feature importance analysis revealed **tenure, order count, and cashback** as top contributors.

3️⃣ **Customer Segmentation**
   - **Cluster 1**: High-value churners—previously frequent buyers but now inactive.
   - **Cluster 2**: Low-value churners—sporadic shoppers with low order counts.
   - **Cluster 3**: Dormant users—signed up but made minimal purchases.

---

## 🔎 Final Conclusion & Recommendations

### 📌 Key Takeaways
✔️ **Personalized Offers**: Target high-value churners with exclusive discounts.

✔️ **Loyalty Programs**: Introduce cashback incentives for at-risk customers.

✔️ **Engagement Strategies**: Enhance user experience with targeted marketing campaigns.

By leveraging machine learning and segmentation, businesses can proactively reduce churn and improve customer retention strategies.
