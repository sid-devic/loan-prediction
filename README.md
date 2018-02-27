# loan-prediction
Loan prediction website/project

To load the saved model, need to:
import sklearn
import pickle

loaded/_model = pickle.load(open('model.sav', 'rb'))
loaded/_model.predict(someDataframe)

We use a scikitlearn logistic model to fit a loan default dataset and predict, given a variety of inputs, whether a given person will default on a loan or not.
