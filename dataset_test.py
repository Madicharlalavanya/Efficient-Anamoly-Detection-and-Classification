import pickle

le = pickle.load(open("models/label_encoder.pkl", "rb"))

print(le.classes_)