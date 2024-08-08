
# Predicting Social Health Insurance Fund Contributions from Kenya’s Informal Sector Workers 
## Table of Contents
1. [Introduction](#introduction)
2. [Business Understanding](#business-understanding)
3. [Data Understanding](#data-understanding)
4. [Problem Statement](#problem-statement)
5. [Objectives](#objectives)
6. [Metric of Success](#metric-of-success)
7. [Data Preparation and Cleaning](#data-preparation-and-cleaning)
8. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
9. [Statistical Analysis](#statistical-analysis)
10. [Modeling](#modeling)
11. [Evaluation](#evaluation)
12. [Conclusions](#conclusions)
13. [Recommendations](#recommendations)
14. [Next Steps](#next-steps)

## Introduction
In 2023, the Kenyan government enacted the Social Health Insurance Act, aiming to achieve Universal Health Coverage (UHC). This initiative strives to ensure that all citizens have access to quality healthcare services without incurring catastrophic health expenses. However, predicting the appropriate contributions for informal sector workers remains a challenge due to their variable incomes. Developing an accurate income prediction model can enhance the SHIF program by ensuring fair contributions, identifying low-income households for targeted social programs, and supporting the most vulnerable.

![Kenya-xs](https://github.com/user-attachments/assets/fec89a87-d9bf-4781-ba55-5c4867674455)


## Business Understanding.
The primary goal of this project is to develop a model that predicts the contributions of informal sector workers to the Social Health Insurance Fund (SHIF). This involves analyzing household demographics, location, income group, and type of work to understand their impact on income predictions.

## Data Understanding
The dataset includes features such as household demographics, location, income groups, and types of work. Initial data exploration involves understanding the structure, identifying missing values, and performing preliminary cleaning.

## Problem Statement
How can we accurately predict the SHIF contributions of informal sector workers to ensure fair and equitable contribution amounts?

## Objectives
### Main Objective
- Develop a model to predict the contribution of informal sector workers to the Social Health Insurance Fund based on household demographics, location, income group, and type of work.

### Specific Objectives
1. Conduct Exploratory Data Analysis (EDA) to understand the distribution and relationships of various features and identify patterns associated with income.
2. Create an easy-to-use chatbot that allows informal sector workers to determine their required contributions to the SHIF based on the prediction model.
3. Utilize insights from the model to inform and support policy decisions, ensuring that SHIF contributions are fair and equitable for all informal sector workers.

## Metric of Success
- Accuracy and F1 Score metrics will be used to evaluate the balance between precision and recall in classifying low-income households within the informal sector, ensuring targeted social programs are appropriately directed.

## Data Preparation and Cleaning
Data preparation involves handling missing values, normalizing features, and encoding categorical variables to ensure the dataset is suitable for model training. Key steps include:
- Dropping irrelevant columns.
- Imputing missing values.
- Encoding categorical variables.

## Exploratory Data Analysis (EDA)
### Key Insights
- **Income Distribution**: 

![respondent_education_vs_income_group](https://github.com/user-attachments/assets/a3da5ef0-9195-457b-9b7d-572a4d880471)


The bar chart illustrates the frequency distribution of income groups based on how much individuals were paid in the last month. The income groups are divided into ranges: 0-10k, 10k-20k, 20k-30k, 30k-40k, and 40k-50k. The chart reveals that the 0-10k and 10k-20k income groups have the highest frequencies, indicating a large portion of the population falls within these lower income brackets. This distribution suggests a concentration of individuals in the lower income ranges.

- **Region Distribution**:

![region_distribution](https://github.com/user-attachments/assets/de9d2214-6478-4b7a-be34-f3e3cbc290a6)

The bar chart depicts the distribution of regions where individuals in Kenya are working in the informal sector. Kisumu has the highest frequency of individuals, followed by West Pokot, Kericho, and Tana River. This visualization highlights the diverse geographic spread of informal sector employment in Kenya.

- **Literacy Levels**:
![literacy_vs_income_group](https://github.com/user-attachments/assets/eb1dcba7-84fa-49d7-85ca-44dbb441781a)

The bar chart illustrates the distribution of literacy levels among individuals. The "able to read whole sentence" category has the highest frequency, suggesting a relatively high level of literacy among the population.

- **Occupation Distribution**:

![occupation_grouped_distribution-1](https://github.com/user-attachments/assets/6697704a-d33c-47a4-aea9-6783c779c709)


The bar chart shows the distribution of various grouped occupations. 'Agriculture - employee' and 'skilled manual' categories have the highest frequencies, indicating a significant portion of the population is employed in agricultural work and skilled manual labour.


- **Education vs Income**:

![respondent_education_vs_income_group-1](https://github.com/user-attachments/assets/937bc17f-4e82-43da-9599-b4f0d9791b18)

The bar chart illustrates the relationship between respondent education levels and income groups. Higher education is associated with higher income groups, while lower education is more common in lower income groups.

- **Age distribution vs academic level**:
![age_age](https://github.com/user-attachments/assets/5d10d360-b6f5-4585-a4e8-1341a97b6bb9)

The swarm plot shows the distribution of respondents' ages across different education levels. Most respondents are between 20 and 45 years old. Categories like complete primary and incomplete secondary have a higher density of respondents, indicating they are more common, while higher education and complete secondary have fewer respondents.

## Statistical Analysis
Statistical methods, such as the Kruskal-Wallis test, are employed to analyze the relationships between different variables and their impact on income predictions. Results indicate significant income differences across various demographic and socio-economic factors, highlighting the multifaceted nature of income determinants.

## Modeling
### Models Used
1. Naive Bayes
2. Logistic Regression
3. Decision Tree
4. Support Vector Machine
5. Random Forest
6. Gradient Boosting
7. K-Nearest Neighbors
8. Deep Learning

## Evaluation 

**Model performance**

![image](https://github.com/user-attachments/assets/53b27fb3-cf8c-44d5-a5bb-95df042c37f4)

The curve plots the True Positive Rate (sensitivity) against the False Positive Rate (1-specificity) for each classifier. From the ROC curve, it is evident that all classifiers except Naive Bayes achieve perfect performance with an Area Under the Curve (AUC) of 1.00. This indicates flawless discrimination between classes for KNN, Decision Tree, Random Forest, Gradient Boosting, SVM, and Logistic Regression models. The Naive Bayes classifier, while still performing well, has a slightly lower AUC of 0.96, indicating some degree of misclassification compared to the other models. The near-perfect and identical ROC curves for KNN, Decision Tree, Random Forest, Gradient Boosting, SVM, and Logistic Regression suggest these models are exceptionally effective for this classification task, reflecting their high accuracy and reliability. The Naive Bayes classifier, although slightly less accurate, still demonstrates strong performance, making it a viable option depending on the specific context and requirements of the task.

**Best Model (K-Nearest Neighbors)**

Models are evaluated using metrics such as accuracy, precision, recall, and F1 score. The confusion matrices provide a visual representation of model performance.
The bar graph indicates that the Random forest model shows outstanding performance. The model achieved perfect classification across all classes, resulting in an overall accuracy of 99 per cent.

## Findings
- **Income Determinants**: Significant income differences were found across various demographic and socio-economic factors, such as age, region, educational attainment, literacy, and household characteristics.
- **Model Performance**: The KNN model demonstrated perfect classification, with an accuracy of 1.00. Other classifiers also achieved high performance, indicating their effectiveness in predicting income groups.
- **Literacy and Income Group**: Higher literacy is associated with higher income groups, while lower literacy is more common in the lowest income groups.
- **Occupation and Income Group**: The majority of individuals in the informal sector are engaged in agricultural work and skilled manual labour.

## Recommendations
1. **Targeted Educational Programs**: Implement programs to improve literacy and vocational skills among informal sector workers.
2. **Tailored Social Programs**: Develop social programs targeting vulnerable groups identified in the study.
3. **Policy Development**: Use data insights to inform policy decisions addressing the unique needs of informal sector workers.
4. **Enhanced Data Collection**: Improve data collection processes to capture comprehensive information on informal sector workers.

## Limitations
1. **Model Generalizability**: Further validation is needed to ensure the models' applicability to other contexts or datasets.
2. **Unaccounted Variables**: Certain socio-economic factors that could influence income levels may not have been captured.
3. **Static Analysis**: Continuous updates and validations of the model are necessary to maintain accuracy.
4. **Data Quality and Completeness**: The study relied on available data, which might have gaps or inaccuracies.

## Conclusions
The study successfully developed a robust model to predict the contributions of informal sector workers to the Social Health Insurance Fund. The findings highlighted significant income disparities influenced by various socio-economic factors and underscored the importance of targeted educational and social programs to support vulnerable groups. Ongoing efforts to improve data quality and capture additional variables will be essential to maintain and enhance the model's effectiveness. The insights gained from this study can inform policy decisions and contribute to a more equitable and efficient SHIF program.

## Future Model Enhancement
1. **Model Enhancement and Deployment**:
   - Fine-tune hyperparameters and experiment with advanced algorithms to improve model accuracy.
   - Integrate the model into a user-friendly interface for easy access.
2. **Data Improvement**:
   - Regularly update the dataset to maintain model relevance.
   - Implement robust data cleaning techniques to handle missing values and outliers.
3. **Policy Recommendations**:
   - Use model insights to identify and support low-income households.
   - Adjust SHIF contribution levels based on predicted incomes to ensure fairness.

## Next Steps
1. **Model Refinement**: Continue refining and optimizing the models for better performance.
2. **Comprehensive Evaluation**: Evaluate models using additional metrics to ensure robustness.
3. **User Engagement**: Develop and deploy user-friendly tools for wider accessibility.
4. **Ongoing Data Collection**: Establish continuous data collection mechanisms to keep the model updated.

By following these recommendations and addressing the identified limitations, this project can significantly contribute to the effectiveness and fairness of Social Health Insurance Fund contributions in Kenya’s informal sector.

Link to Tableau Public: https://public.tableau.com/app/profile/cynthia.dalmas/viz/Understanding_Kenyan_population/Dashboard1 

## How to deploy 
 On the terminal run python app.py 
 
Run the URL on the browser
