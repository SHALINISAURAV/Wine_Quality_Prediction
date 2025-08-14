# 🍷 Wine Quality Prediction

## 📌 Project Overview
The **Wine Quality Prediction** project is a machine learning application that predicts the quality of wine based on various physicochemical properties. Using a dataset of wine samples, the model learns patterns that distinguish high-quality wines from lower-quality ones.

This project involves:
- Data preprocessing
- Exploratory data analysis (EDA)
- Feature selection
- Model training and evaluation
- Saving and reusing the trained model

---

## 📂 Dataset
- **File**: `WineQT.csv`
- **Source**: Public wine quality dataset
- **Description**: Contains various chemical properties of wine such as acidity, sugar content, pH, and alcohol percentage, along with a quality rating.

**Sample Columns:**
- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality` (target variable)

---

## 🛠️ Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib & Seaborn** (Data Visualization)
- **Scikit-learn** (Model Building & Evaluation)
- **Jupyter Notebook**

---

## 🚀 Installation & Usage
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/wine-quality-prediction.git
   cd wine-quality-prediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook Wine_Quality_Prediction.ipynb
   ```

4. **Load and Use the Saved Model**
   ```python
   import joblib
   model = joblib.load('wine_quality_model.pkl')
   prediction = model.predict([[7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]])
   print("Predicted Wine Quality:", prediction)
   ```

---

## 📊 Model Evaluation
- **Accuracy**: Achieved high accuracy on the test dataset
- **Confusion Matrix**: Visualized for performance analysis
- **Classification Report**: Precision, Recall, and F1-Score for each class

---

## 📌 Future Improvements
- Optimize hyperparameters for better performance
- Implement other ML algorithms like Random Forest, XGBoost
- Deploy as a web application

---

## 📜 License
This project is licensed under the MIT License.

---

## 👩‍💻 Author
**Shalini Saurav**
- GitHub: [ShaliniSaurav](https://github.com/ShaliniSaurav)

