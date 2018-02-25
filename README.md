# loan-prediction
HackUTD Loan prediction website/project

To load the saved model, need to:
import sklearn
import pickle

loaded/_model = pickle.load(open('model.sav', 'rb'))
loaded/_model.predict(someDataframe)

