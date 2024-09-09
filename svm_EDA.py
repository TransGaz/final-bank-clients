import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


def get_svm_model_prediction(random_state: int=42):
    """ Функция которая предсказывает отклик клиентов согласно модели опрных векторов SVM"""

    RANDOM_STATE = random_state
    DATASET_PATH = 'datasets/bank_clients.csv'

    df = pd.read_csv(DATASET_PATH)
    X = df.drop(['TARGET'], axis=1)
    y = df['TARGET']

    # Обучим SVM с линейным ядром на тренировочной выборке.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    svm_calibrated = CalibratedClassifierCV(SVC(kernel='linear', probability=True), cv=3)
    svm_calibrated.fit(X_train, y_train)
    probs_svm_c = svm_calibrated.predict_proba(X_test)[:, 1]

    return  probs_svm_c


