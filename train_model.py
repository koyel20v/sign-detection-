from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

X = np.load("data.npy")
y = np.load("labels.npy")

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained!")