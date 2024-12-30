# Soiligator: Soil Analysis and Irrigation Prediction

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>

![Screenshot 2024-12-30 at 10 54 24 PM](https://github.com/user-attachments/assets/0232ec7f-a77f-4cee-8984-6928530c6a4a)

## Overview
**Soiligator** is an advanced machine learning project designed to optimize irrigation management by predicting whether irrigation is necessary based on environmental and soil-related data. Leveraging feature engineering and robust predictive models, Soiligator provides actionable insights that improve agricultural efficiency and sustainability.

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>

## Key Features
- **Predictive Models**: Utilizes Logistic Regression, Random Forest, and Support Vector Machine (SVM) algorithms for accurate irrigation predictions.
- **Feature Engineering**: Incorporates non-linear interaction terms and outlier handling for enhanced model performance.
- **Scalable Design**: Easily extendable to include additional features like soil type and crop variety.
- **Data Resilience**: Designed to handle label noise and outliers, ensuring robustness in real-world applications.

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Description](#data-description)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results](#results)
8. [Future Work](#future-work)

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>

## Installation
To use this project, install the required Python packages with the following command:

```bash
pip install -r requirements.txt
```

### Key Dependencies:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning model training and evaluation

Alternatively, install the libraries manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>

## Usage

### Data Loading
Start by loading the dataset `modified_irrigation_dataset.csv`, which includes:
- **Moisture**: Soil moisture content.
- **Temperature**: Ambient temperature.
- **Humidity**: Air humidity level.
- **Irrigation_Needed**: Target label indicating whether irrigation is required.

### Running the Code
The implementation is available in a Jupyter Notebook: `soil_analysis.ipynb`. Execute the cells sequentially to:
1. Load and preprocess the dataset.
2. Engineer additional features.
3. Train machine learning models.
4. Evaluate and compare model performance.

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>

## Data Description
The dataset comprises features representing soil and environmental conditions:
- **Moisture**: Measures the water content in the soil (0–100%).
- **Temperature**: Ambient temperature in degrees Celsius.
- **Humidity**: Air humidity as a percentage (0–100%).

### Engineered Features:
- **Moisture_Temp_Interaction**: Interaction term between soil moisture and temperature to capture non-linear effects.
- **Humidity_Squared**: Non-linear transformation of humidity to account for atmospheric retention properties.

### Data Challenges:
- **Outliers**: Synthetic outliers introduced in 5% of the data to test model resilience.
- **Label Noise**: Added noise to 5% of target labels to simulate real-world conditions.

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>

## Model Training and Evaluation

### Preprocessing
- **Outlier Handling**: Removes or neutralizes extreme values.
- **Feature Scaling**: Standardizes features using `StandardScaler` for optimal model performance.
- **Train-Test Split**: Splits the data into 80% training and 20% testing subsets.

### Models Used:
1. **Logistic Regression**: A baseline model for binary classification.
2. **Random Forest Classifier**: An ensemble learning model for handling complex patterns.
3. **Support Vector Machine (SVM)**: A robust classifier for high-dimensional data.

### Evaluation Metrics:
- **Accuracy**: Overall correctness of predictions.
- **Confusion Matrix**: Breakdown of true positives, false positives, true negatives, and false negatives.
- **ROC Curve and AUC Score**: Measures the model's ability to distinguish between classes.
- **Precision-Recall Curve**: Highlights performance in handling imbalanced data.
- **Classification Report**: Includes precision, recall, F1-score, and support.

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>

## Results
### Model Comparison:
- **Logistic Regression**: Achieved baseline performance with moderate accuracy.
- **Random Forest**: Outperformed other models, achieving high accuracy and robustness to noise and outliers.
- **SVM**: Demonstrated strong performance on standardized features but required longer training times.

### Visualization:
- **Confusion Matrix**: Provided for each model to analyze prediction errors.
- **ROC Curves**: Highlighted the trade-offs between sensitivity and specificity.
- **Precision-Recall Curves**: Demonstrated model effectiveness on imbalanced datasets.

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>

## Future Work
1. **Hyperparameter Tuning**: Optimize models using Grid Search or Random Search to improve accuracy.
2. **Feature Expansion**: Include additional predictors such as:
   - Soil type
   - Crop type
   - Real-time weather forecasts
3. **Time-Series Analysis**: Incorporate temporal data to predict irrigation needs over time.
4. **Deployment**: Package the model into a web or mobile application for practical use by farmers and agricultural experts.

---

## Contribution
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For any queries, feel free to contact the project owner.

<p align="left">
  <img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="1920" 
</p>
