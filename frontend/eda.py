import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@st.cache_data
def read_json(path="https://finalbankmarketing.onrender.com/api/clients"):
    """ Функция читающая данные с json  """

    DATASET_PATH = "datasets/bank_clients_api.csv"

    df = pd.read_csv(path, index_col='id')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates()
    df = df.set_index('agreement_rk')
    df.to_csv(DATASET_PATH)
    return df

@st.cache_data
def read_data(path="datasets/bank_clients_clean.csv"):
    """ Функция читающая данные с csv файла """

    df = pd.read_csv(path)
    return df

@st.cache_data
def get_rows_number(df: pd.DataFrame) -> int:
    """ Функция вычисляющая число сторк в датафрейме  """
    row_number = df.shape[0]

    return row_number


def drop_dupliсates(df: pd.DataFrame) -> pd.DataFrame:
    """ Функция вычисляющая число сторк в датафрейме """
    new_df = df.drop_duplicates()

    return new_df


def get_positive_response(df: pd.DataFrame) -> int:
    """ Функция вычисляющая число полученных откликов """
    filtered_df = df[df['TARGET'] == 1]
    response = filtered_df.shape[0]

    return response


def find_min_value(df: pd.DataFrame, column: str) -> str:
    """ Функция вычисляющая минимальные значения  колонки датафрейма"""
    min_value = float(df[column].min())
    round(min_value, 2)

    return str(min_value)


def find_max_value(df: pd.DataFrame, column: str) -> str:
    """ Функция вычисляющая максимальные значения колонки датафрейма """

    max_value = df[column].max()
    round(max_value, 2)

    return str(max_value)


def find_mean_value(df: pd.DataFrame, column: str) -> str:
    """ Функция вычисляющая среднее значения колонки датафрейма """
    mean_value = df[column].mean()
    round(mean_value, 2)

    return str(mean_value)


def train_data(file_path: str = 'datasets/bank_clients.csv', test_size: float = 0.25, random_state: int = 42):
    """Функция  разбивающие тренировочные данные"""

    df = pd.read_csv(file_path)

    X = df.drop(['TARGET', 'AGREEMENT_RK'], axis=1)
    y = df['TARGET']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_val, y_train, y_val


def show_metrics(model, limit: float, df: pd.DataFrame) -> str:
    """ Функция  отображающие метрики  модели"""
    text = ""

    X_train, X_val, y_train, y_val = train_data()

    model.fit(X_train, y_train)
    pred_test = model.predict(X_val)

    probs = model.predict_proba(X_val)
    probs_churn = probs[:, 1]

    classes = probs_churn > limit

    text += f"\nAccuracy: {accuracy_score(y_val, classes)}\n"
    text += f"\nPrecision: {precision_score(y_val, classes)}\n"
    text += f"\nRecall: {recall_score(y_val, classes)}\n"
    text += f"\nf1: {f1_score(y_val, classes)}\n"

    return text
