import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset into a pandas DataFrame
data = pd.read_csv('data/fact_d.csv')

# Encode your categorical features using LabelEncoder
le_city = LabelEncoder()
le_product = LabelEncoder()
le_agent = LabelEncoder()
le_city.fit(data['city'].unique())
le_product.fit(data['product'].unique())
le_agent.fit(data['agent'].unique())
data['city'] = le_city.fit_transform(data['city'])
data['product'] = le_product.fit_transform(data['product'])
data['agent'] = le_agent.fit_transform(data['agent'])

# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['city', 'product']], data['agent'], test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the performance of the trained model on the test data
accuracy = clf.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.3f}')

# Save the trained model to a joblib file
joblib.dump(clf, 'model.joblib')
joblib.dump(le_city, 'le_city.joblib')
joblib.dump(le_product, 'le_product.joblib')
joblib.dump(le_agent, 'le_agent.joblib')
