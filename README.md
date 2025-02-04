## Forest Cover Prediction

### Overview
This project focuses on predicting forest cover types using machine learning techniques. The dataset contains various geographical and environmental features, and the goal is to classify each instance into one of the forest cover types.

---

### Data Loading and Exploration
#### Importing Required Libraries
The following libraries are used:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
```

#### Loading the Dataset
The dataset is loaded using pandas:

```python
data = pd.read_csv('train.csv')
data.head()
```

- `train.csv` contains various features related to forest cover, including elevation, slope, soil type, and wilderness areas.
- The dataset is inspected using `.head()`, `.describe()`, and `.dtypes` to understand its structure.

#### Data Cleaning and Preprocessing

```python
df = data.drop('Id', axis='columns')
df.sample(5)
```

- The `Id` column is removed since it does not contribute to classification.
- A sample of the dataset is displayed to verify the structure.

---

### Feature Scaling
Min-Max Scaling is applied to normalize the feature values:

```python
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df.drop('Cover_Type', axis=1))
y = df['Cover_Type']
```

- **MinMaxScaler**: Transforms feature values to a range of [0,1], improving model performance.
- **Target Variable (`Cover_Type`)**: Extracted separately for classification.

---

### Train-Test Split
The dataset is split into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(df_scaled, y, test_size=0.2, random_state=42)
```

- **80% training data** and **20% test data** ensure robust evaluation.
- `random_state=42` ensures reproducibility.

---

### Model Training
Three machine learning models are used:

#### Random Forest Classifier

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

- **RandomForestClassifier**: An ensemble method that builds multiple decision trees and averages their predictions.

#### Gradient Boosting Classifier

```python
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
```

- **GradientBoostingClassifier**: A boosting algorithm that sequentially improves model performance by correcting errors from previous iterations.

#### Logistic Regression

```python
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
```

- **LogisticRegression**: A simple, interpretable classification model.

#### Voting Classifier (Ensemble Learning)

```python
voting_clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='hard')
voting_clf.fit(X_train, y_train)
```

- **Voting Classifier**: Combines multiple classifiers to improve overall performance.

---

### Model Evaluation
The trained models are evaluated on the test set using accuracy and classification reports.

```python
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

- **Accuracy Score**: Measures overall correctness of predictions.
- **Classification Report**: Provides precision, recall, and F1-score for each class.

---

### Conclusion
This project demonstrates the effectiveness of ensemble learning for forest cover classification. The Voting Classifier enhances accuracy by leveraging multiple models. Further improvements can be achieved by fine-tuning hyperparameters and exploring additional features.

