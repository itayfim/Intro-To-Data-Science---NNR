import time
import json
import pandas as pd

from sklearn.preprocessing import StandardScaler
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score

def read_and_scale_data(data):
    df = pd.read_csv(data)
    y = df['class']
    x = df.drop(columns=['class'])
    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x))
    return x_scaled, y


def find_optimal_radius(x_valid_scaled, x_train_scaled, y_train, y_valid):
    optimal_radius, radius = 0, 0
    max_accuracy, accuracy = 0, 0
    mode = y_train.mode()[0]  # calculating the mode of class column
    end_time = time.time() + 295  # 300 seconds is five minutes, so taking 5 sec for the rest of the calculations
    while time.time() < end_time and max_accuracy - accuracy < 0.1: # if there is massive reduction in the accuracy - ending the loop to save spare (wasted) runtime
        predictions = []
        for i, vld_inst in x_valid_scaled.iterrows():
            distances = np.linalg.norm(x_train_scaled - vld_inst, axis=1) # caclculating the distances
            nearest_neighbors = y_train[distances <= radius]
            if nearest_neighbors.empty:
                predictions.append(mode)
            elif len(nearest_neighbors) > 0:
                prediction = max(set(nearest_neighbors), key=list(nearest_neighbors).count)
                predictions.append(prediction)
        accuracy = accuracy_score(y_valid, predictions)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_radius = radius
        print(optimal_radius, radius, accuracy, max_accuracy)
        radius += 0.01
    return optimal_radius


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')
    x_train_scaled, y_train = read_and_scale_data(data_trn)
    x_valid_scaled, y_valid = read_and_scale_data(data_vld)
    x_test_scaled, y_test = read_and_scale_data(data_tst)

    optimal_radius = find_optimal_radius(x_valid_scaled, x_train_scaled, y_train, y_valid)
    mode = y_train.mode()[0]  # calculating the mode of class column

    predictions = []
    for i, tst_inst in x_test_scaled.iterrows():
        distances = np.linalg.norm(x_train_scaled - tst_inst, axis=1)
        nearest_neighbors = y_train[distances <= optimal_radius]
        if nearest_neighbors.empty:
            predictions.append(mode)
        elif len(nearest_neighbors) > 0:
            prediction = max(set(nearest_neighbors), key=list(nearest_neighbors).count)
            predictions.append(prediction)

    return predictions

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert (len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time() - start, 0)} sec')
