import pandas as pd
import tkinter as tk
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#dataset
df = pd.read_csv('loan.csv', skipinitialspace=True)

#features and target variable
X = df[['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
        'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']]
y = df['loan_status']

#training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                                       'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                                       'bank_asset_value']),
        ('categorical', OneHotEncoder(drop='first'), ['education', 'self_employed'])
    ])
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(random_state=42))])

# Train
model.fit(X_train, y_train)

#Graphical User Interface
class LoanPredictionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Loan Status Prediction GUI")

        
        self.feature_entries = {}
        for column in X.columns:
            label = tk.Label(master, text=column.replace('_', ' ').title())
            entry = tk.Entry(master)
            label.pack()
            entry.pack()
            self.feature_entries[column] = entry

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.result_label = tk.Label(master, text="Prediction: ")

        
        self.predict_button.pack()
        self.result_label.pack()

    def get_user_input(self):
        return pd.DataFrame([{column: entry.get() for column, entry in self.feature_entries.items()}])

    def predict(self):
       
        user_input = self.get_user_input()
        prediction = model.predict(user_input)
        self.result_label.config(text=f"Prediction: {prediction[0]}")

root = tk.Tk()
loan_prediction_gui = LoanPredictionGUI(root)
root.mainloop()
