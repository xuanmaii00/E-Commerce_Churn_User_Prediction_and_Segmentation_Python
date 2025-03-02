# E-Commerce Churn User Prediction and Segmentation_Python

## 📌 Table of Contents

- [Background & Overview](#background--overview)
- [Dataset Description & Data Structure](#dataset-description--data-structure)
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

## ⚒️ Main Process

### 1️⃣ Import Necessary Libraries

- **Data Handling:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Preprocessing:** LabelEncoder, StandardScaler  
- **Machine Learning:** RandomForestClassifier, KMeans, AgglomerativeClustering  
- **Evaluation:** metrics  
- **Dimensionality Reduction:** PCA  

### 2️⃣ Exploratory Data Analysis (EDA)

#### Handling Data Issues:
- Identified and addressed missing or null values.
- Removed duplicate records.
- Ensured correct data types for analysis.

![Image](https://github.com/user-attachments/assets/c8e6bd85-1607-4267-a36c-d6e4976c38c1)

![Image](https://github.com/user-attachments/assets/289ecbf3-c4cc-4806-959e-4fa72de273ed)

-> The dataset has an imbalanced class distribution, with significantly more non-churned users than churned users.

-> This imbalance needs to be addressed when applying the Random Forest model to prevent bias towards the majority class and improve the prediction of churned users.

#### Feature Analysis:
- Univariate analysis for distributions of numerical and categorical variables.
- Bivariate analysis to determine correlations with churn.

![Image](https://github.com/user-attachments/assets/3f376b98-19bd-46cd-90e6-6064e9e3fb97)

-> Churn has a weak positively correlation with Complain(0.23), a weeak negative correlation with Tenure(-0.33)

-> This means users who complain more tend to churn more and user who have been using the app for a long time tend not to churn.
#### Key Findings:
- Customers with fewer orders and lower engagement had a higher churn rate.
- Longer distance from warehouse to home increased churn probability.
- Payment preferences influenced churn behavior.

### 3️⃣ Supervised Model Training: Random Forest

#### Preprocessing Steps:
- Categorical variables encoded using Label Encoding.
- Applied SMOTE to balance class distribution.

![Image](https://github.com/user-attachments/assets/0da9bbb0-2bae-4841-8788-587d09e51c76)

![Image](https://github.com/user-attachments/assets/163ce168-3b34-4002-b6a2-86b45f93ede9)

#### Model Performance:
- Accuracy: **98%**
- Feature importance: **Tenure, Order Count, Cashback, and Warehouse Distance** were top predictors.

![Image](https://github.com/user-attachments/assets/72750d2c-9433-4d3a-9a52-1edce8bfd38f)

### 4️⃣ Unsupervised Learning: K-Means Clustering

#### Segmentation Strategy:
- Applied PCA for dimensionality reduction.
![Image](https://github.com/user-attachments/assets/6fc1ca45-43e3-4b7b-995e-8b466ba0042f)
-> 3 intrinsic dimensions -> keep 3 PCA components

  
- Determined 4 optimal clusters using the elbow method.
![Image](https://github.com/user-attachments/assets/d7bd836e-768b-4e71-ac85-10afdea9c19d)
--> We will choose k=4


- Used t-SNE for visualization to evaluate model performance
![Image](https://github.com/user-attachments/assets/a27e5fe3-7055-4ea2-b1b7-aa90984d8e7d)
-> Well separted groups



---

## 📊 Key Insights & Visualizations

### 1️⃣ **Churn Drivers**
![Image](https://github.com/user-attachments/assets/4b5b2964-d218-467e-9d44-92361ca1cd8a)
The following features have been identified as important in predicting churn:

Tenure (24.3%), Marital Status (Single: 9%), Day Since Last Order (6.8%), Warehouse to Home (5.5%), Cashback Amount (4.9%), Coupon Used (4.2%), Order Count (4.2%), Hour Spend on App (3.2%)

![Image](https://github.com/user-attachments/assets/8c18d4e9-7d7d-44f8-b6e3-d8f1cd2ae6e1)
-> Users with a shorter duration of engagement with the platform are more likely to churn

-> New customers might not be finding enough value or satisfaction within their initial experience. Targeted onboarding processes or engagement strategies could help retain these users.

![Image](https://github.com/user-attachments/assets/9a7e7e26-1d09-4ba5-a73a-7eaf2ebbe259)
-> The significance of marital status suggests that single users may have a higher likelihood of churning compared to married users

-> Need to understanding the preferences and behaviors of single users to tailor marketing strategies or offers that resonate with this demographic.

![Image](https://github.com/user-attachments/assets/5a621171-eea8-44ba-8090-8982923cabc1)
-> Users who have recently placed orders may be more likely to churn, indicating that they might not find sufficient value in the service or product offered after their initial engagement.

![Image](https://github.com/user-attachments/assets/fe8a8c66-cd2a-429f-b9ea-67b1855b3fcb)
-> Churned users tend to have their home futher from the Warehouse since distance from the warehouse to the customer's home can significantly affect delivery speed and customer satisfaction, thereby influencing their likelihood to churn

-> Reducing delivery times by optimizing warehouse locations or enhancing logistics strategies for customers living farther away may help improve customer satisfaction and retention

![Image](https://github.com/user-attachments/assets/eeace3a1-c389-4ca1-8882-a114eed13903)
-> Churned users tend to receive less cash back amount

![Image](https://github.com/user-attachments/assets/d588a8ae-263d-402b-95b8-917225df7e6a)
-> The percentage of churned users who use coupon is lower suggesting that users who do not utilize these promotions are more likely to churn

-> Offering better incentives or personalized promotions could enhance user retention, particularly for those who show low engagement with existing offers.

![Image](https://github.com/user-attachments/assets/cb82c285-5546-431f-a028-6b3f445278f4)
-> Users spending longer hours on the app tend to have a higher churn rate, suggeting that they might need to spend too much time finding the products they need

-> Enhancing user experience to dencrease time spent on the app through improved search functionality, user-friendly navigation, personalized recommendations, and streamlined checkout processes can help users find what they need more efficiently .

### 2️⃣ **Churn Segments Characteristics**

![Image](https://github.com/user-attachments/assets/5f9f2dbf-4668-448c-bb5b-1c7c770f89b5)

![Image](https://github.com/user-attachments/assets/26517ef1-8f8f-474f-8264-ebfb1135032e)

Cluster 0: This cluster has a median cashback amount of approximately 175, making it the second highest among the clusters. The values range from around 140 to 230, and the distribution is symmetrical with no significant outliers.

Cluster 1: Featuring the highest median cashback amount at around 200, Cluster 1 also displays the greatest variability, as indicated by a larger interquartile range (IQR). Values extend from about 150 to over 230, with no significant outliers present.

Cluster 2: With the lowest median cashback amount near 140, Cluster 2 shows little variance. However, it does include multiple outliers that stretch up to about 200.

Cluster 3: Similar to Cluster 2, this cluster has a median close to 150, which is the third highest. It contains multiple outliers, with some cashback amounts reaching up to 230, while the majority of values remain concentrated in the lower range.

![Image](https://github.com/user-attachments/assets/be2c8ea8-0cbf-4f13-9fb2-75e69004a1a3)
-> The number of churned users who prefer to order mobile phones increases from Cluster 0 to Cluster 3, with Cluster 3 exhibiting a distinctively high number of preferred orders for mobile phones.

![Image](https://github.com/user-attachments/assets/63f5b262-2cc1-4af9-a250-69a1ba68c9f2)
Cluster 1 has the highest median tenure at around 7, along with the highest variance (ranging from 1 to 20), indicating that churned users in this cluster tend to have longer tenures with the company, showcasing a diverse range of experiences.

Cluster 0 has the second-highest median tenure at approximately 2, with a variance from 1 to 10 and some higher outliers. This suggests that while 75% of users have relatively short tenures (below 4), there is a notable 25% of long-term users in this cluster, with tenures ranging from above 4 to around 10.

Cluster 2 and Cluster 3 both exhibit low median tenures, around 1 and 2, respectively. Their small interquartile range (IQR) indicates that users in these clusters are relatively new to the company.

![Image](https://github.com/user-attachments/assets/10b1e46f-015d-4485-b018-13531a9f9628)
Cluster 1 has the highest median order count, around 7, and the largest interquartile range (IQR), indicating a broad spread in order counts. 75% of users in this cluster place between 1 and 8 orders, while the top 25% place between 8 and 15. This suggests Cluster 1 users are highly active, with many making frequent purchases.

Cluster 0 has the second-highest median order count, about 2. 75% of users place between 1 and 3 orders, with the remaining 25% placing between 3 and 6 orders. This indicates moderate purchasing activity.

Cluster 2 and Cluster 3 have relatively low median order counts. Cluster 2 users typically place between 1 and 3 orders, while Cluster 3 users mostly place around 2 orders. These clusters represent users with lower purchasing activity compared to Cluster 0 and Cluster 1.

Overall, there are clear differences in ordering behavior across clusters, with Cluster 1 being the most active and Cluster 2 and Cluster 3 representing lower activity levels.

![Image](https://github.com/user-attachments/assets/296ce8b6-6fca-4b71-be56-8e03567886b2)
Cluster 0 has the highest overall number of laptop and accessory orders, showing a distinctively high number of laptop orders compared to the other clusters. Following this are Cluster 2, Cluster 1, and Cluster 3, with the number of orders decreasing in that order.

Overall, the chart indicates that laptop and accessory orders are most popular in Cluster 0, with a clear preference among users for placing orders in this category. This suggests that Cluster 0 users have a strong inclination towards purchasing laptops and accessories compared to users in other clusters.

![Image](https://github.com/user-attachments/assets/14654b11-8ecd-4047-bd78-4be51b6a9e09)
Cluster 0 is dominated by users from CityTier 3, with a very high count compared to other city tiers. There is a small presence of CityTier 1 users and virtually no CityTier 2 users.

Cluster 1 has a balanced mix between CityTier 1 and CityTier 3, with CityTier 1 having a slightly higher count. CityTier 2 is barely present.

Cluster 2 is similar to Cluster 1, with users primarily from CityTier 1 but also some presence from CityTier 3. CityTier 2 again has almost no representation.

Cluster 3 is heavily dominated by CityTier 1 users, with a significant count. CityTier 3 has a much smaller representation, while CityTier 2 has a very low count.

Overall, CityTier 3 is most prominent in Cluster 0, while CityTier 1 dominates Cluster 3. CityTier 2 has very few users across all clusters.

---

## 🔎 Final Conclusion & Recommendations

### 📌 Key Takeaways

✔️ **Personalized Offers**: Target high-value churners with exclusive discounts.

✔️ **Loyalty Programs**: Introduce cashback incentives for at-risk customers.

✔️ **Engagement Strategies**: Enhance user experience with targeted marketing campaigns.

