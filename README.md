# Churn User Prediction and Segmentation_Python
Developed a Random Forest model to predict churn and a KMeans model to segment churned users into distinct behavior-based groups, providing actionable insights into general churn behaviors and patterns within each group

## I. Introduction

### 1. Business Objective
In the competitive landscape of e-commerce, retaining customers is vital for long-term success. This project aims to address the challenge of customer churn by predicting which users are likely to leave the platform. The specific objectives are as follows:

- **Predict Churned Users**: Develop a supervised machine learning model to predict users who are likely to churn (leave the platform). For this, we utilize the Random Forest algorithm, which is effective in handling complex datasets and capturing nonlinear relationships.

- **Identify Churn Patterns**: Analyze the patterns and behaviors of churned users to understand the underlying factors contributing to their decision to leave. Insights from this analysis will provide actionable recommendations to the company for reducing churn rates.

- **Segment Churned Users**: In addition to prediction, we aim to create an unsupervised learning model using K-Means clustering. This model will classify churned users into distinct behavioral groups, enabling the company to tailor promotions and marketing strategies effectively for each segment.

### 2. Project Overview
This project employs two main approaches:

- **Supervised Learning**: 
  - **Model**: Random Forest
  - **Objective**: Predict which users are likely to churn based on historical data.
  - **Outcome**: A predictive model that can identify at-risk users and provide insights for retention strategies.

- **Unsupervised Learning**:
  - **Model**: K-Means Clustering
  - **Objective**: Classify churned users into distinct behavioral groups based on their interaction data.
  - **Outcome**: Identification of specific user behaviors that can inform targeted promotional offers, enhancing user retention.

### 3. Key Contributions
- Developed and fine-tuned machine learning models to predict user churn and segment churned users.
- Provided actionable insights and recommendations for improving customer retention strategies.
- Enabled the e-commerce company to implement tailored promotions based on user behavior, potentially reducing churn rates.

### 4. Dataset Dictionary
The dataset used for this project contains several variables related to customer behavior and interactions with the e-commerce platform. Below is a description of each variable:

| **Variable**                  | **Description**                                                                              |
|-------------------------------|----------------------------------------------------------------------------------------------|
| **CustomerID**                 | Unique customer ID.                                                                          |
| **Churn**                      | Churn flag indicating whether the customer has churned (1) or not (0).                       |
| **Tenure**                     | The number of months the customer has been with the company.                                |
| **PreferredLoginDevice**       | The preferred device used by the customer to log into the platform (e.g., mobile, desktop).    |
| **CityTier**                   | City tier (1, 2, or 3): geographic classification of the customer’s location.                  |
| **WarehouseToHome**            | Distance between the warehouse and the customer’s home address.                              |
| **PreferredPaymentMode**       | Preferred payment method used by the customer (e.g., credit card, PayPal).                    |
| **Gender**                     | Gender of the customer (Male/Female).                                                        |
| **HourSpendOnApp**             | Number of hours the customer spends on the app or website.                                   |
| **NumberOfDeviceRegistered**   | The total number of devices registered under the customer’s account.                         |
| **PreferredOrderCat**          | The category of products the customer orders most frequently in the last month.              |
| **SatisfactionScore**          | The customer’s satisfaction score based on their recent experiences with the platform.        |
| **MaritalStatus**              | Marital status of the customer (e.g., Single, Married).                                      |
| **NumberOfAddress**            | Total number of addresses associated with the customer.                                      |
| **Complain**                   | Whether the customer raised any complaints in the last month (Yes/No).                       |
| **OrderAmountHikeFromLastYear**| The percentage increase in the customer’s order amount compared to the previous year.        |
| **CouponUsed**                 | Total number of coupons used by the customer in the last month.                              |
| **OrderCount**                 | The total number of orders placed by the customer in the last month.                         |
| **DaySinceLastOrder**          | The number of days since the customer placed their last order.                               |
| **CashbackAmount**             | The average cashback amount the customer received in the last month.                         |

## II/ Project Coding

### 1. Import Necessary Libraries
The following libraries are imported for various purposes:

- **Data Handling**: `pandas`, `numpy` - Used for data manipulation and analysis.
- **Visualization**: `matplotlib`, `seaborn` - Help in visualizing data distributions and relationships.
- **Preprocessing**: `LabelEncoder`, `StandardScaler` - Used for scaling numerical data and encoding categorical variables into numerical format.
- **Machine Learning Models**: `RandomForestClassifier`, `KMeans`, `AgglomerativeClustering` - Used for classification and clustering tasks.
- **Evaluation**: `metrics` - Tools like confusion matrix and classification report are used to evaluate model performance.
- **Dimensionality Reduction**: `PCA` - Used to reduce the dimensions of the data for clustering.

These libraries facilitate the cleaning, transforming, modeling, and evaluating of the dataset throughout the project.

### 2. Exploratory Data Analysis (EDA)
The first step in any data science project is to perform exploratory data analysis (EDA) to understand the dataset. This includes:

- **Handling Missing Data**: Identifying and addressing missing or null values in the dataset.
- **Removing Duplicates**: Ensuring there are no duplicate rows that could affect the analysis.
- **Correcting Data Types**: Ensuring each column has the correct data type for analysis (e.g., numerical, categorical).

#### Univariate Analysis
- **Numerical Data**: Basic statistics like mean, median, and standard deviation are used to understand the distribution of numerical variables such as `Tenure`, `OrderCount`, etc.
- **Categorical Data**: Distribution of categorical variables like `Gender`, `CityTier`, and `PreferredPaymentMode` is visualized to see their distribution.

#### Bivariate and Multivariate Analysis
- We analyze relationships between `Churn` (our target variable) and other features. For example:
  - **Correlation with Churn**: 
    - `Churn` has a weak positive correlation with `Complain` (0.23), meaning users who complain are slightly more likely to churn.
    - `Churn` has a weak negative correlation with `Tenure` (-0.33), indicating that longer-tenured users are less likely to churn.

These findings suggest that complaints significantly impact churn, while customers who have been on the platform longer tend to stay.

### 3. Supervised Model Training: Random Forest
Now, we apply the **Random Forest model** for predicting churn:

#### Data Preprocessing
- **Transform Categorical Variables**: Convert categorical columns like `Gender` and `PreferredPaymentMode` into numerical format using label encoding.
- **Balancing Classes with SMOTE**: Since the dataset is imbalanced (more non-churned users than churned), we apply **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the churn classes, ensuring better model performance.

#### Model Training
- Split the dataset into training and test sets using **train_test_split**.
- Apply the **Random Forest model** on the training set to predict churn.
- Evaluate the model’s performance using **confusion matrix** and **classification report**.

The model achieves excellent results:
- **Accuracy**: 98% of churned and non-churned users are predicted correctly.
- **Feature Importance**: The model identifies the top features contributing to churn, including:
  - **Tenure** (24.3%)
  - **Marital Status** (Single: 9%)
  - **Day Since Last Order** (6.8%)
  - **Warehouse to Home** (5.5%)
  - **Cashback Amount** (4.9%)
  - **Coupon Used** (4.2%)
  - **Order Count** (4.2%)
  - **Hour Spend on App** (3.2%)

**The detailed analysis and insights are performed in the code file uploaded.**

## 4. Unsupervised Model: K-Means Clustering
Next, we apply **K-Means clustering** to segment churned users into distinct behavioral groups for targeted promotions.

### Dimensionality Reduction with PCA
To reduce the dimensions of the data and avoid overfitting, we apply **Principal Component Analysis (PCA)**. PCA helps in reducing the complexity of the data while preserving its essential structure. We determine that 3 PCA components capture the intrinsic dimensions of the data effectively.

### Standard Scaling
We scale the features using **StandardScaler** to ensure that the data has zero mean and unit variance. This step is crucial for improving the performance of clustering algorithms like K-Means.

### Determining the Optimal Number of Clusters
We use the **elbow method** to choose the optimal number of clusters. The elbow method helps identify the point where adding more clusters doesn’t significantly improve the model’s performance. In our case, we determine that 4 clusters are optimal.

### Clustering with K-Means
We apply **K-Means clustering** to partition churned users into 4 distinct clusters. To evaluate the clustering performance:
- **t-SNE Visualization**: We use **t-SNE** to project the high-dimensional data into 2D for visualization. The 2D graph shows reasonably well-separated clusters, indicating effective segmentation.
- **Silhouette Score**: The silhouette score of 0.4 indicates a moderately well-defined clustering structure, with distinct and reasonably separated groups.

### Feature Importance for Clusters
After clustering, we apply the **Random Forest model** to predict the cluster label using the churned user data. The model identifies the top 6 features that distinguish the clusters. These features provide deeper insights into the behavioral differences between the clusters. 
**The detailed analysis and insights are performed in the code file uploaded.**




