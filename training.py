import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import joblib
data = pd.DataFrame({
            "region": ['north', 'south', 'north', 'south', 'north'],
            "age": [20, 35, 18, 27, 24],
            "gender": ['male', 'female', 'female', 'male','male'],
            "budget": [4000, 2500, 1850, 3785, 5100],
            "previous_purchase": ['bag', 'smartphone',  'shoes', 'watch', 'laptop'],
            "product": ['shoes', 'wig', 'iphone', 'tshirt', 'bag'],  

})
x = data.drop(columns=['product'], axis=1)
y = data['product']

ct = ColumnTransformer(
        transformers=[('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['region', 'gender', 'previous_purchase'])],
        remainder='passthrough'
)

new_x = ct.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(new_x, y, test_size=0.3, random_state=42)

model = LogisticRegression(
          C=1,
          class_weight='balanced',
          max_iter=100,
          penalty='l1',
          solver='liblinear'
)


#,model = DecisionTreeClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

print(f"accuracy {accuracy} ")

#save model

joblib.dump(model, "sales_model.pkl")
#save encoder
joblib.dump(ct, "sales_encoder.pkl")
print("model and encoder saved")

region = input("state your region: ")
age = input("input your age: ")
gender = input("state your gender: ")
budget = input("what is your budget: ")
previous_purchase = input("what is your previous purchase: ")
sample_data = pd.DataFrame({
                           "region": [region],
                           "age": [age],
                           "gender": [gender],
                           "budget": [budget],
                           "previous_purchase": [previous_purchase]
})

converted_sample_data = ct.transform(sample_data)

recomendation = model.predict(converted_sample_data)

print("recomended: ", recomendation[0])