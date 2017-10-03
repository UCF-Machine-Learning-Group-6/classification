import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.special import logsumexp, expit

def load_csv(filename):
    dataset = pd.read_csv(filename, names = ['fixed acidity', 'volatile acidity', 'citric acid', 
                                             'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'])
    return dataset

def kfold(dataset):
    kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    return kf

def sigmoid(scores):
    return expit(scores)

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target*scores - np.log(1 + np.exp(scores)) )
    return ll

def accuracy(testX, testY, weights):
    acc = 0
    index = 0
    final_scores = np.dot(testX, weights)
    preds = np.round(sigmoid(final_scores))
    for value in preds:
        if value == testY[index]:
            acc = acc + 1
        index = index + 1
    return acc / len(preds)

def classify(trainX, trainY, testX, testY):
    weights = np.zeros(trainX.shape[1])
    for step in range(300000):
        scores = np.dot(trainX, weights)
        predictions = sigmoid(scores)
        output_error_signal = trainY - predictions
        gradient = np.dot(trainX.T, output_error_signal)
        weights += 5e-5 * gradient
    return accuracy(testX, testY, weights)   
    
    

if __name__ == "__main__":
    np.random.seed(12)
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    simulated_data = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))
    
    bias = np.ones((simulated_data.shape[0], 1))
    simulated_data = np.hstack((bias, simulated_data))
    
    acc = classify(simulated_data, simulated_labels, simulated_data, simulated_labels)
    print("Synthetic data accuracy:" + str(acc))

    filename = "Final_wine.csv"
    dataset = load_csv(filename)
    folds = kfold(dataset)
    category = 3
    cataccuracy = 0

    for quality in dataset['quality'].unique():
        totalaccuracy = 0
        print ("Quality: " + str(category))
        category = category + 1
        run = 1
        
        for train_index, test_index in folds.split(dataset):
            print("Run: " + str(run))
            trainY = []
            testY = []
            
            trainX = np.asarray([[row['volatile acidity'], row['chlorides'], row['total sulfur dioxide'], row['sulphates'], row['alcohol']] for index, row in dataset.iloc[train_index].iterrows()])            
            bias = np.ones((trainX.shape[0], 1))
            trainX = np.hstack((bias, trainX))
            
            for index, row in dataset.iloc[train_index].iterrows():
                if row['quality'] == quality:
                    trainY.append(1)
                else:
                    trainY.append(0)
            
            for index, row in dataset.iloc[test_index].iterrows():
                if row['quality'] == quality:
                    testY.append(1)
                else:
                    testY.append(0)
            
            trainY = np.asarray(trainY)
            testY = np.asarray(testY)
            
            testX = np.asarray([[row['volatile acidity'], row['chlorides'], row['total sulfur dioxide'], row['sulphates'], row['alcohol']] for index, row in dataset.iloc[test_index].iterrows()])
            bias = np.ones((testX.shape[0], 1))
            testX = np.hstack((bias, testX))
           
            runaccuracy = classify(trainX, trainY, testX, testY)
            
            print ('Run Accuracy: ' + str(runaccuracy))
            totalaccuracy = totalaccuracy + runaccuracy
            
            if (run == 5):
                print("Total Accuracy: " + str(totalaccuracy/5))
                cataccuracy = cataccuracy + (totalaccuracy / 5)
            
            run = run+1
    
    print("Total Accuracy of all categories: " + str(cataccuracy/6))
