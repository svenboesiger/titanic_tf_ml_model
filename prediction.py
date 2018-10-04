import tensorflow as tf
import pandas as pd

classifier = None


def initialize_classifier():
    passenger_features = [tf.feature_column.numeric_column(key='pclass'),
                          tf.feature_column.numeric_column(key='age'),
                          tf.feature_column.numeric_column(key='sibsp'),
                          tf.feature_column.numeric_column(key='parch'),
                          tf.feature_column.numeric_column(key='sex_male'),
                          tf.feature_column.numeric_column(key='sex_female'),
                          tf.feature_column.numeric_column(key='embarked_C'),
                          tf.feature_column.numeric_column(key='embarked_Q'),
                          tf.feature_column.numeric_column(key='embarked_S')]

    global classifier
    classifier = tf.estimator.DNNClassifier(
        hidden_units=[20, 20, 20],
        feature_columns=passenger_features,
        model_dir='ml_model/titanic',
        n_classes=2)


def post(passengar):
    if classifier is None:
        initialize_classifier()

    input_data = pd.DataFrame.from_records([passengar['attributes']])
    predictions = classifier.predict(input_fn=tf.estimator.inputs.pandas_input_fn(
        x=input_data,
        shuffle=False))

    survival_probability = int((next(predictions)['probabilities'][0] * 100)), 201

    return survival_probability