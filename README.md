# 🏠 California Housing Price Prediction using Machine Learning

A complete end-to-end Machine Learning project that analyzes the **California Housing Prices** dataset, performs data preprocessing and exploratory data analysis (EDA), builds multiple regression models, compares their performance, and deploys the best-performing model as an interactive Streamlit web application.

---

## 🚀 Live Demo

🌐 **Streamlit Application:**  
https://bjtbj3gywt25mazsckyoyp.streamlit.app/

---

## 📌 Project Overview

The goal of this project is to predict **California house prices** using machine learning techniques. The project follows the complete data science lifecycle, including:

- Data Import & Inspection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Simple Linear Regression
- Multiple Linear Regression
- Random Forest Regression
- Hyperparameter Tuning
- Feature Importance Analysis
- Interactive Streamlit Dashboard

The project demonstrates how data preprocessing, visualization, and machine learning can be combined to solve a real-world regression problem.

---

# 📂 Dataset Information

**Dataset:** California Housing Prices

**Target Variable**
- `median_house_value`

**Features Used for Final Model**
- Median Income
- Housing Median Age
- Latitude
- Longitude
- Households

The dataset contains housing information collected from California census districts, including geographical, demographic, and economic characteristics.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Joblib
- Streamlit
- Statsmodels

---

# 📁 Project Structure

```text
.
├── app.py                             # Streamlit application
├── Week1_Task.ipynb                   # Jupyter Notebook
├── california_housing_streamlit.csv   # Dataset
├── random_forest_model.pkl            # Trained Random Forest model
├── requirements.txt                   # Required libraries
└── README.md
```

---

# 📖 Project Workflow

The notebook is organized into the following sections:

### Part 1 – Data Import & Inspection
- Imported required libraries
- Loaded the dataset
- Explored dataset structure
- Identified numerical and categorical features
- Generated descriptive statistics

### Part 2 – Data Cleaning
- Checked missing values
- Filled missing values using the median
- Removed duplicate records
- Verified data types
- Checked for invalid values

### Part 3 – Exploratory Data Analysis (EDA)
- Summary statistics
- Histograms
- Box plots
- Scatter plots
- Correlation Heatmap
- Pair Plot
- Outlier Analysis
- Skewness Analysis

### Part 4 – Insights & Observations
- Distribution analysis
- Correlation analysis
- Feature relationships
- Outlier interpretation

### Part 5 – Simple Linear Regression
- Feature selection
- Model training
- Model evaluation
- Residual analysis

### Part 6 – Multiple Linear Regression
- Multiple feature selection
- Performance comparison
- Coefficient interpretation

### Part 7 – Visualization Portfolio
Professional visualizations including:
- Histograms
- Box Plots
- Scatter Plots
- Correlation Heatmap
- Regression Line
- Residual Plot

### Bonus Features
- Random Forest Regression
- Hyperparameter Tuning (Randomized Search)
- Permutation Feature Importance
- Interactive Streamlit Dashboard

---

# ⚙️ How to Run the Notebook

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

## 2. Navigate to the Project Folder

```bash
cd <repository-name>
```

## 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook (`Week1_Task.ipynb`) and run all cells sequentially.

---

# 🌐 Run the Streamlit Application

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

After running the command, the application will open automatically in your browser.

---

# 📦 Required Libraries

```text
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
joblib
streamlit
statsmodels
```

Or install them using:

```bash
pip install -r requirements.txt
```

---

# 🤖 Machine Learning Models

The following regression models were developed and evaluated:

- Simple Linear Regression
- Multiple Linear Regression
- Random Forest Regressor (Hyperparameter Tuned)

### Performance Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The Random Forest Regressor achieved the best predictive performance among all models.

---

# 📊 Key Findings

1. **Median Income** is one of the strongest predictors of California house prices, showing a strong positive relationship with the target variable.

2. **Latitude** and **Longitude** significantly influence house prices, indicating that geographical location plays a major role in determining property values.

3. The dataset contains several outliers, particularly in count-based features such as Total Rooms, Population, and Households, highlighting the importance of data exploration before modeling.

4. The target variable (`median_house_value`) has a noticeable cap at **$500,001**, suggesting that house prices were truncated in the original dataset.

5. Multiple Linear Regression outperformed Simple Linear Regression by utilizing multiple relevant features simultaneously.

6. The tuned **Random Forest Regressor** produced the highest prediction accuracy by effectively capturing complex, non-linear relationships within the data.

7. Permutation Feature Importance revealed that **Latitude**, **Longitude**, and **Median Income** are the most influential features in predicting California house prices.

---

# 📈 Visualizations Included

- Histogram
- Box Plot
- Scatter Plot
- Correlation Heatmap
- Pair Plot
- Regression Line Plot
- Residual Plot
- Feature Importance Chart

---

# 💻 Streamlit Dashboard Features

The deployed web application allows users to:

- Predict California house prices interactively
- Adjust housing feature values using sliders
- View feature importance
- Explore correlation heatmaps
- Visualize feature distributions
- Preview the dataset

---

# 🎯 Project Outcome

This project demonstrates a complete machine learning workflow, from raw data to a deployed predictive application. It highlights the importance of data cleaning, exploratory analysis, feature engineering, model evaluation, and deployment in building reliable machine learning solutions. The final Random Forest model provides accurate house price predictions, while the Streamlit dashboard makes the model accessible through an interactive web interface.

---

# 👨‍💻 Author

**Ayush Kumar Verma**

Machine Learning Assignment – California Housing Price Prediction

---
**Live Application:** https://bjtbj3gywt25mazsckyoyp.streamlit.app/
