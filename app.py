import streamlit as st
import pickle
import pandas as pd

from PIL import Image

from frontend.eda import read_data, get_rows_number, get_positive_response, find_max_value, find_min_value, find_mean_value
from frontend.eda import train_data, show_metrics
from svm_EDA import get_svm_model_prediction


def preload_content():
    """ предварительная загрузка содержимого, используемого в веб-приложении """
    bank_img = Image.open('images/bank.png')
    age_img = Image.open('images/age_img.png')
    targhet_img = Image.open('images/targhet_img.png')
    income_img = Image.open('images/income_img.png')
    correlation_img = Image.open('images/correlation_img.png')
    targhet_socstatus_img = Image.open('images/targhet_socstatus_img.png')
    loanss_img = Image.open('images/loanss_img.png')
    gender_img = Image.open('images/gender_img.png')

    return bank_img, age_img, targhet_img, income_img, correlation_img, targhet_socstatus_img, loanss_img, gender_img


def render_page(bank_img, age_img, targhet_img, income_img, correlation_img, targhet_socstatus_img, loanss_img, gender_img):
    st.title('Удовлетворённость обслуживанием БАНКА')
    st.subheader('Делаем деньги, предсказываем удовлетворённость, оцениваем важность факторов')
    st.write('Материал - отклики клиентов')
    st.image(bank_img)

    tab1, tab2, tab3, tab4 = st.tabs([':mag: Исследовать', 'Предсказать LR', 'Предсказать LR modified', 'Предсказать SVM'])

    with tab1:
        st.write('EDA: исследуем наши данные, предварительно очищенные и обработанные :sparkles:')

        #data = read_json()
        #time.sleep(20)

        #if data is None:
            #df = read_data()
        #else:
            #df = data
        df = read_data()
        df = df.drop_duplicates()

        st.write('Исследование проводилось на ' + str(get_rows_number(df)) + ' респондентов.')
        st.write(' Наше предложение получило ' + str(get_positive_response(df)) + ' обратных откликов')

        st.image(targhet_img)

        st.divider()

        st.write('**Возраст респодентов**')
        age_column = 'AGE'
        st.image(age_img)
        st.write('Возраст  респондентов варьирует  от ' + find_min_value(df, age_column))
        st.write('до ' + find_max_value(df, age_column) + ' лет.')
        st.write('Средний возраст респондентов: ' + find_mean_value(df, age_column) + ' лет.')

        st.divider()

        st.write('**Доходы респодентов**')
        income_column = 'PERSONAL_INCOME'
        st.image(income_img)

        st.write('Доходы  респондентов варьирует  от ' + find_min_value(df, income_column)+ ' руб')
        st.write('до ' + find_max_value(df, income_column) + ' руб.')
        st.write('Средний доход респондентов: ' + find_mean_value(df, income_column)+ ' руб.')

        st.divider()

        st.write('**Гистограмма социального статуса респондентов**')
        st.image(targhet_socstatus_img)
        st.write('Наблюдается превалирование работающих респондентов и активных пенсионеров над неработающими')
        st.divider()

        st.write('**Гистограмма кол-во займов взятых и закрытых**')
        st.image(loanss_img)
        st.write('Большинство респондентов имеют по 1-2 кредита, которые благополучно оплатили')
        st.divider()

        st.write('**Гистограмма откликов по гендорной принадлежности**')
        st.image(gender_img)
        st.write('Наблюдается явное приемущество мужского пола в проводимых исследованиях.')
        st.divider()

        st.write('**Матрица кореляции**')
        st.write(df.corr())
        st.image(correlation_img)

        st.divider()
    with tab2:
        st.header(" Standart Linear Regression")

        with open('datasets/model_lr.pickle', 'rb') as f:
            model = pickle.load(f)

        limit = st.slider("Выберите порог для перевода предсказаний модели в классы", 0.0, 1.0, 0.01)
        st.write(show_metrics(model, limit, df))
        st.divider()

        st.write("Давайте посмотрим, с какой вероятностью разные клиенты ответили бы на нашу рассылку!")
        option1 = st.selectbox('Выберите клиента', tuple([i for i in range(15594)]))
        X_train, X_val, y_train, y_val = train_data()
        df_new = pd.concat([X_train, X_val], axis=0)
        df_new.reset_index(inplace=True, drop=True)
        prob = model.predict_proba(df_new)[option1][1]
        st.write(f"Клиент откликнется на маркетинговое предложение с вероятностью {round(prob, 2) * 100}%")

    with tab3:
        st.header(" Special Linear Regression, C=1.2, max_iter=5000")

        with open('datasets/model_mlr.pickle', 'rb') as f:
            model = pickle.load(f)

        limit = st.slider("Выберите порог для модели LR special", 0.0, 1.0, 0.01)
        st.write(show_metrics(model, limit, df))
        st.divider()

        st.write("Давайте посмотрим, с какой вероятностью разные клиенты ответили бы на нашу рассылку!")
        option2 = st.selectbox('Выберите номер клиента', tuple([i for i in range(15594)]))
        X_train, X_val, y_train, y_val = train_data()
        df_new = pd.concat([X_train, X_val], axis=0)
        df_new.reset_index(inplace=True, drop=True)
        prob = model.predict_proba(df_new)[option2][1]
        st.write(f"Клиент откликнется на маркетинговое предложение с вероятностью {round(prob, 2) * 100}%")

    with tab4:
        st.header(" SVM")

        st.divider()
        st.write("Давайте посмотрим, с какой вероятностью разные клиенты ответили бы на нашу рассылку!")
        option3 = st.selectbox('Выберите номер клиента', tuple([i for i in range(15594)]), key="<uniquevalueofopt3>")

        st.divider()
        result = get_svm_model_prediction()
        # TO DO
        # depriciated version
        probab_svm = result[option3]

        st.write(f"Клиент {option3} откликнется на маркетинговое предложение с вероятностью {round(probab_svm, 2) * 100}%")
        st.divider()
def load_page():
    """ loads main page """

    bank_img, age_img, targhet_img, income_img, correlation_img, targhet_socstatus_img, loanss_img, gender_img = preload_content()

    st.set_page_config(layout="wide",
                       page_title="Банки и опросы",
                       page_icon=':bank:')

    render_page(bank_img, age_img, targhet_img, income_img, correlation_img, targhet_socstatus_img, loanss_img, gender_img)


if __name__ == "__main__":
    load_page()
