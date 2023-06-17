import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, render_template, request

# Load the dataset
df = pd.read_csv("fraudTrain.csv", nrows=5000)

# Create the feature matrix and target vector
X = df.drop(["is_fraud"], axis=1)
y = df["is_fraud"]

# Convert categorical variables to dummy variables
X = pd.get_dummies(X)

# Train the model using Random Forest algorithm
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Saving the trained model
filename = 'fraud_detection_model2.sav'
pickle.dump(rf, open(filename, 'wb'))

# Load the trained model
loaded_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        amount = float(request.form['amount'])
        category = request.form['category']
        state = request.form['state']

        # Create a DataFrame with user input
        new_data = pd.DataFrame({'name': [name], 'gender': [gender], 'amount': [amount],
                                 'category': [category], 'state': [state]})

        # Verify the features
        training_features = X.columns
        new_data_features = new_data.columns

        # Handle missing features
        missing_features = set(training_features) - set(new_data_features)
        for feature in missing_features:
            new_data[feature] = 0  # Add missing features with default value

        # Reorder the features to match the training data
        new_data = new_data[training_features]

        # Convert categorical variables to dummy variables
        new_data = pd.get_dummies(new_data)

        # Make predictions
        predictions = loaded_model.predict(new_data)

        if predictions[0] == 0:
            result = "The transaction is not fraudulent."
        else:
            result = "The transaction is fraudulent!"

        return render_template('result.html', result=result)

    return render_template('ClassifierForm.html')

if __name__ == '__main__':
    app.run()
