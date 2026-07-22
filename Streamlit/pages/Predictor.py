import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
from xgboost import XGBClassifier
from pathlib import Path

def calculate_values(df):
    days = df['days_since_prior_order'].astype(int)
    basket = df['basket_values'].astype(int)

    days_since_prior_order = days.iloc[-1]
    Previous_orders_number = len(df.index)
    average_days_between_orders = round(days.sum() / len(days), 2)
    average_of_3_last_orders = round(days[-3:].sum() / 3, 2)

    min_interval = min(days)
    max_interval = max(days)
    interval_std = round(stat.stdev(days),2)

    basket_size = basket.iloc[-1]
    average_basket_size = round(basket.sum() / len(basket), 2)
    basket_ratio = round((len(basket)* basket.iloc[-1] )/ basket.sum(), 2)

    data = [1, Previous_orders_number, days_since_prior_order, average_days_between_orders, average_of_3_last_orders, max_interval, min_interval, basket_size, average_basket_size, basket_ratio, 0.5, 4, 17,interval_std]

    return data

def input_data():
    st.write("### Input Data")

    st.write("##### Number of prior orders")
    number_inputs = st.number_input('Number of Prior orders', step=1, min_value=1)
    st.write('Number of Prior orders', number_inputs)

    data = pd.DataFrame(columns=['days_since_prior_order', 'basket_values'])

    if 'df' not in st.session_state:
        st.session_state.df = data

    col1, col2 = st.columns(2)
    input_values = [col1.text_input(f'Days since last order', i+1, key=f"text_input_{i}")
            for i in range(number_inputs)]
    basket_values = [col2.text_input('basket value', 5, key=f"basket_input_{i}")
            for i in range(number_inputs)]

    if st.button("Add to df", key="button_update"):
        data.drop(data.index, inplace=True)
        st.session_state.df = data

        # Update dataframe state
        st.session_state.df = pd.concat(
            [st.session_state.df, pd.DataFrame({'days_since_prior_order': input_values, 'basket_values': basket_values})],
            ignore_index=True)
        st.text("Updated dataframe")

    "## Dataframe state:"
    st.dataframe(st.session_state.df)
    return st.session_state.df

def display_info(data):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Days since prior order: " + str(data[2]))
        st.write("Number of previous orders: " + str(data[1]))
        st.write("Average number of days between orders: " + str(data[3]))
        st.write("Average number of days between 3 last orders: " + str(data[4]))

    with col2:
        st.write("Minimum days since prior order: " + str(data[6]))
        st.write("Maximum days since prior order: " + str(data[5]))
        st.write("Standard Deviation: " + str(data[13]))

    with col3:
        st.write("Basket size of last order: " +  str(data[7]))
        st.write("Average basket size: " + str(data[8]))
        st.write("Basket size compared to average: " +  str(data[9]))


def present_results(pred, prob):
    st.write("### Results")
    day_range = ''

    if pred == 0:
        day_range = '0-4 days'
    elif pred == 1:
        day_range = '5-7 days'
    elif pred == 2:
        day_range = '8-14 days'
    else:
        day_range = '15+ days'


    st.metric(label="Prediction", value = day_range)

    st.write("##### Confidence")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="0-4 days", value = f"{prob[0]*100:,.2f}%")
    col2.metric(label="5-7 days", value = f"{prob[1]*100:,.2f}%")
    col3.metric(label="8-14 days", value = f"{prob[2]*100:,.2f}%")
    col4.metric(label="15+ days", value = f"{prob[3]*100:,.2f}%")


def main():
    script_location = Path(__file__).absolute().parent.parent.parent
    st.title("Calculate user return")

    df = input_data()
    try:
        data = calculate_values(df)
        display_info(data)

        model = XGBClassifier()
        model.load_model(script_location / "Models/xgboost_model.json")
        pred = model.predict([data])
        prob = model.predict_proba([data])[0]
        present_results(pred,prob)
    except IndexError:
        st.write("## **Insert at least two rows to make a prediction**")
    except stat.StatisticsError:
        st.write("## **Insert at least two rows to make a prediction**")


if __name__ == "__main__":
    main()









