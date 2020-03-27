#Runs all the models through the accuracy/time reporting process in baselineModels.ipynb
#so we don't have to repeat a bunch of code 
#Should take in trained models 

from statistics import mean 
from sklearn.model_selection import cross_validate

#y must be a pandas series, X must be a pandas dataframe 
#models must be trained sklearn models 

def hugeFramework(modelSVM, modelLR, modelNN, modelRF, X, y):
    SVM_cv_results = cross_validate(modelSVM, X, y, cv=3) 
    LR_cv_results = cross_validate(modelLR, X, y, cv=3)
    NeuralNet_cv_results = cross_validate(modelNN, X, y, cv=3)
    RF_cv_results = cross_validate(modelRF, X, y, cv=3)
    
    print("Accuracies:")
    print("SVM_cv_results: " + str(mean(SVM_cv_results['test_score'])))
    print("LR_cv_results: " + str(mean(LR_cv_results['test_score'])))
    print("NeuralNet_cv_results: " + str(mean(NeuralNet_cv_results['test_score'])))
    print("RandomForest_cv_results: " + str(mean(RF_cv_results['test_score'])))
    print()
    
    print("Training times:")
    print("SVM_cv_results: " + str(mean(SVM_cv_results['fit_time'])))
    print("LR_cv_results: " + str(mean(LR_cv_results['fit_time'])))
    print("NeuralNet_cv_results: " + str(mean(NeuralNet_cv_results['fit_time'])))
    print("RandomForest_cv_results: " + str(mean(RF_cv_results['fit_time'])))
    print()
        
    print("Prediction times:")
    print("SVM_cv_results: " + str(mean(SVM_cv_results['score_time'])))
    print("LR_cv_results: " + str(mean(LR_cv_results['score_time'])))
    print("NeuralNet_cv_results: " + str(mean(NeuralNet_cv_results['score_time'])))
    print("RandomForest_cv_results: " + str(mean(RF_cv_results['score_time'])))