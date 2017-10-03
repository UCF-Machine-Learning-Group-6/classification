import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn import metrics

from sklearn.cross_validation import train_test_split



import plotly.graph_objs as go

import plotly.plotly as py

from plotly.graph_objs import *

py.sign_in('SashankMLAI', 'T6j3atXP4H36Ws4Zzlza')



# Dataset Path

DATASET_PATH = "Final_Wine.csv"





def scatter_with_color_dimension_graph(feature, target, layout_labels):

    trace1 = go.Scatter(

        y=feature,

        mode='markers',

        marker=dict(

            size='16',

            color=target,

            colorscale='Viridis',

            showscale=True

        )

    )

    layout = go.Layout(

        title=layout_labels[2],

        xaxis=dict(title=layout_labels[0]), yaxis=dict(title=layout_labels[1]))

    data = [trace1]

    fig = Figure(data=data, layout=layout)

    # plot_url = py.plot(fig)

    py.image.save_as(fig, filename=layout_labels[1] + '_Density.png')





def create_density_graph(dataset, features_header, target_header):

    for feature_header in features_header:

        print ("Creating density graph for feature:: {} ".format(feature_header))

        layout_headers = ["Number of Observation", feature_header + " & " + target_header,

                          feature_header + " & " + target_header + " Density Graph"]

        scatter_with_color_dimension_graph(dataset[feature_header], dataset[target_header], layout_headers)





def main():

    wine_data_headers = ["Id", "fixed.acidity", "volatile.acidity", "citric.acid","residual.sugar", "chlorides", "free.sulfur.dioxide","total.sulfur.dioxide", "density", "pH", "sulphates", "alchhol", "quality"]

    wine_data = pd.read_csv(DATASET_PATH, names=wine_data_headers)



    print ("Number of observations :: ", len(wine_data.index))

    print ("Number of columns :: ", len(wine_data.columns))

    print ("Headers :: ", wine_data.columns.values)

    print ("Target :: ", wine_data[wine_data_headers[-1]])

 



    train_x, test_x, train_y, test_y = train_test_split(wine_data[wine_data_headers[:-1]],

                                                        wine_data[wine_data_headers[-1]], train_size=0.8)

    # Train multi-classification model with logistic regression

    lr = linear_model.LogisticRegression()

    lr.fit(train_x, train_y)



    # Train multinomial logistic regression model

    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)



    print ("Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, lr.predict(train_x)))

    print ("Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, lr.predict(test_x)))



    print ("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))

    print ("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))





if __name__ == "__main__":

    main()
