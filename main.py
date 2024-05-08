import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

url = 'https://drive.google.com/file/d/1FyLouCoWaUqrVlMOTb4lGCBGHs_YNfOg/view?usp=sharing'
df = pd.read_csv('https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2]), sep = ';')


st.title('')
st.write('Выберите входные данные:')


#выбор самолета: выбор рейса и даты вылета
flt_nums = [1120, 1122, 1124, 1126, 1128, 1130, 1132]
flt_num = st.selectbox('Выберите номер рейса', flt_nums)

# min_date = datetime(2018, 1, 1)
# max_date = datetime(2018, 12, 31)
selected_date = st.selectbox('Выберите дату вылета в 2018 году', 
                              df[df.FLT_NUM == flt_num].DD.unique())
selected_date = pd.to_datetime(selected_date)
df['DD'] = pd.to_datetime(df['DD'])

#наш конкретный самолет:
df_plane = df[(df['FLT_NUM']==flt_num) & (df['DD']==selected_date)].drop_duplicates()
#st.write(df_plane)
st.title('Исторические данные')
#отрисовка суммы PASS_BK по номеру самолета FLT_NUM
pivot_table_FLT_NUM = pd.pivot_table(df_plane,values='PASS_BK', 
                                     index='DTD', columns=['FLT_NUM'], aggfunc=np.sum)
fig_FLT_NUM_selected_date = px.line(pivot_table_FLT_NUM, title='График спроса по конкретному самолету')
fig_FLT_NUM_selected_date.update_layout(xaxis=dict(autorange='reversed'))
fig_FLT_NUM_selected_date.update_xaxes(tickvals=[200, 150, 100, 50, -1], 
                                       ticktext=['200', '150', '100', '50', '-1'])
st.plotly_chart(fig_FLT_NUM_selected_date) #можно добавить .update_xaxes(type='log'), чтобы масштаб по иксу логарифмический был
st.write(f'Общее число неявок: {df_plane["NS"].sum()}')

# Filter dataframe
df_filtered = df_plane[df_plane['DTD'] == 0].set_index('SEG_CLASS_CODE')

# Create bar plot
fig = px.bar(df_filtered, x=df_filtered.index, y='PASS_BK', labels={'x':'Класс', 'y':'Спрос'}, title='График спроса по каждому классу данного рейса в день вылета')

# Show the plot
st.plotly_chart(fig)




SSCL1s = ['Y', 'C']
SSCL1 = st.selectbox('Исторические данные по кабине:', SSCL1s)

#отрисовка суммы PASS_BK по кабине SSCL1
pivot_table_SSCL1 = pd.pivot_table(df_plane[df_plane['SSCL1']==SSCL1], values='PASS_BK',
                             index='DTD', columns='SSCL1', aggfunc=np.sum)
fig_SSCL1 = px.line(pivot_table_SSCL1, title=f'График спроса по кабине {SSCL1}')
fig_SSCL1.update_layout(xaxis=dict(autorange='reversed'))
fig_SSCL1.update_xaxes(tickvals=[200, 150, 100, 50, -1], 
                                       ticktext=['200', '150', '100', '50', '-1'])
st.plotly_chart(fig_SSCL1)
st.write(f'Неявки по кабине {SSCL1}:', df_plane[df_plane['SSCL1']==SSCL1]['NS'].sum())


SEG_CLASS_CODEs = ['N', 'E', 'G', 'J', 'L', 'Q', 'T', 'X']
SEG_CLASS_CODE = st.selectbox('Исторические данные по классу:', SEG_CLASS_CODEs)

#отрисовка суммы PASS_BK по классу билета SEG_CLASS_CODE
pivot_table_SEG_CLASS_CODE = pd.pivot_table(df_plane[df_plane['SEG_CLASS_CODE']==SEG_CLASS_CODE], 
                                            values='PASS_BK',index='DTD', 
                                            columns='SEG_CLASS_CODE', aggfunc=np.sum)
fig_SEG_CLASS_CODE = px.line(pivot_table_SEG_CLASS_CODE, 
                             title=f'График спроса по классу {SEG_CLASS_CODE}')
fig_SEG_CLASS_CODE.update_layout(xaxis=dict(autorange='reversed'))
fig_SEG_CLASS_CODE.update_xaxes(tickvals=[200, 150, 100, 50, -1], 
                                ticktext=['200', '150', '100', '50', '-1'])
st.plotly_chart(fig_SEG_CLASS_CODE)

st.title('Прогноз')

# Добавление интерактивного виджета (ползунка)
DTD = st.slider('Выберите количество дней до вылета', 15, 1)
predict_types = ['Весь самолёт', 'Кабина', 'Класс']
predict_type = st.selectbox('Прогноз по:', predict_types)#, key='visibility')
st.write('Выбранное вами число:', DTD)
if predict_type == 'Весь самолёт':
    #st.write(pivot_table_FLT_NUM.reset_index())
    min_diff = 10**10
    data = list(pivot_table_FLT_NUM.reset_index().sort_values(by='DTD', ascending=False)[pivot_table_FLT_NUM.columns[-1]]) * 3
    # for i in SEG_CLASS_CODE:
    #     data1 = list(df_plane[(df_plane.SEG_CLASS_CODE == i)].sort_values(by='DTD', ascending=False).PASS_BK) * 3
    #     if min_diff > abs(data[-1] - data1[-1]):
    #         min_diff = abs(data[-1] - data1[-1])
    #         our_class = i
    # data1 = list(df_plane[(df_plane.SEG_CLASS_CODE == our_class)].sort_values(by='DTD', ascending=False).PASS_BK) * 3
    data1 = list(pd.pivot_table(df_plane[df_plane['SSCL1']=='Y'], values='PASS_BK',
                             index='DTD', columns='SSCL1', aggfunc=np.sum).reset_index().sort_values(by='DTD', ascending=False)['Y']) * 3
    train_data = data[:-DTD]
    test_data = data[-DTD:]
    train_data1 = data1[:-DTD]
    test_data1 = data1[-DTD:]
    mod = sm.tsa.SARIMAX(train_data, exog=train_data1,
                                    order=[2, 2, 2],
                                    seasonal_order=[1, 1, 1, 6],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()

                # Прогнозирование на тестовом наборе данных
    pred = results.forecast(DTD, exog=test_data1)
    df1 = pd.DataFrame({'x': list(range(len(data) // 3 - 2, -2, -1)),
                    'test_data': data[-len(data) // 3:],
                    'pred': data[-len(data) // 3:-DTD]+list((pred / 0.98).astype(int) + 1)})

    # Строим график с помощью plotly.express
    fig_ = px.line(df1, x='x', y=['test_data'], title='График исторических данных и предсказан')
    fig_.add_scatter(x=df1['x'].iloc[-DTD:], y=df1['pred'].iloc[-DTD:], mode='lines', name='pred')
    fig_.update_layout(xaxis=dict(autorange='reversed'))
    fig_.update_xaxes(tickvals=[200, 150, 100, 50, -1], 
                        ticktext=['200', '150', '100', '50', '-1'])
    # Устанавливаем диапазон по умолчанию для оси x (приближение)
    #fig_.update_xaxes(range=[5, 10])

    # Включаем rangeslider для возможности отдаления графика
    #fig_.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_)
    pred = list((pred / 0.98).astype(int) + 1)
    if len(pred) > 2:
        our_ns = pred[-3] - pred[-2]
        if our_ns < 0: our_ns = 0
        st.write(f'Прогноз неявок: {(our_ns // 2)}')
elif predict_type == 'Класс':
    min_diff = 10**10
    data = list(df[(df.FLT_NUM == flt_num) & (df.DD == selected_date) & (df.SEG_CLASS_CODE == SEG_CLASS_CODE)].drop_duplicates(subset='DTD').sort_values(by='DTD', ascending=False).PASS_BK) * 3
    for i in list(set(SEG_CLASS_CODEs) - set([SEG_CLASS_CODE])):
        data1 = list(df[(df.FLT_NUM == flt_num) & (df.DD ==selected_date) & (df.SEG_CLASS_CODE == i)].drop_duplicates(subset='DTD').sort_values(by='DTD', ascending=False).PASS_BK) * 3
        if min_diff > abs(data[-1] - data1[-1]):
            min_diff = abs(data[-1] - data1[-1])
            our_class = i
    data1 = list(df[(df.FLT_NUM == flt_num) & (df.DD ==selected_date) & (df.SEG_CLASS_CODE == our_class)].drop_duplicates(subset='DTD').sort_values(by='DTD', ascending=False).PASS_BK) * 3

    train_data = data[:-DTD]
    test_data = data[-DTD:]
    train_data1 = data1[:-DTD]
    test_data1 = data1[-DTD:]
    mod = sm.tsa.SARIMAX(train_data, exog=train_data1,
                                    order=[2, 2, 2],
                                    seasonal_order=[1, 1, 1, 6],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()

                # Прогнозирование на тестовом наборе данных
    pred = results.forecast(DTD, exog=test_data1)
    df1 = pd.DataFrame({'x': list(range(len(data) // 3 - 2, -2, -1)),
                    'test_data': data[-len(data) // 3:],
                    'pred': data[-len(data) // 3:-DTD]+list(pred.astype(int))})

    # Строим график с помощью plotly.express
    fig_ = px.line(df1, x='x', y=['test_data'], title='График исторических данных и предсказан')
    fig_.add_scatter(x=df1['x'].iloc[-DTD:], y=df1['pred'].iloc[-DTD:], mode='lines', name='pred')
    fig_.update_layout(xaxis=dict(autorange='reversed'))
    fig_.update_xaxes(tickvals=[200, 150, 100, 50, -1], 
                        ticktext=['200', '150', '100', '50', '-1'])
    # Устанавливаем диапазон по умолчанию для оси x (приближение)
    #fig_.update_xaxes(range=[5, 10])

    # Включаем rangeslider для возможности отдаления графика
    #fig_.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_)
elif predict_type == '-':
    ...
else:
    if SSCL1 == 'C':
        min_diff = 10**10
        data = list(pivot_table_SSCL1.reset_index().sort_values(by='DTD', ascending=False)[pivot_table_SSCL1.columns[-1]]) * 3
        for i in SEG_CLASS_CODE:
            data1 = list(df_plane[(df_plane.SEG_CLASS_CODE == i)].sort_values(by='DTD', ascending=False).PASS_BK) * 3
            if min_diff > abs(data[-1] - data1[-1]):
                min_diff = abs(data[-1] - data1[-1])
                our_class = i
        data1 = list(df_plane[(df_plane.SEG_CLASS_CODE == our_class)].sort_values(by='DTD', ascending=False).PASS_BK) * 3
    else:
        data = list(pivot_table_SSCL1.reset_index().sort_values(by='DTD', ascending=False)[pivot_table_SSCL1.columns[-1]]) * 3
        data1 = list(pivot_table_FLT_NUM.reset_index().sort_values(by='DTD', ascending=False)[pivot_table_FLT_NUM.columns[-1]]) * 3
    train_data = data[:-DTD]
    test_data = data[-DTD:]
    train_data1 = data1[:-DTD]
    test_data1 = data1[-DTD:]
    mod = sm.tsa.SARIMAX(train_data, exog=train_data1,
                                    order=[2, 2, 2],
                                    seasonal_order=[1, 1, 1, 6],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()

                # Прогнозирование на тестовом наборе данных
    if SSCL1 == 'Y':
        pred = (results.forecast(DTD, exog=test_data1) / 0.95).astype(int) - 3
    else:
        pred = ((results.forecast(DTD, exog=test_data1) + np.array(test_data)*2)/3 ).astype(int)
        
        
    df1 = pd.DataFrame({'x': list(range(len(data) // 3 - 2, -2, -1)),
                    'test_data': data[-len(data) // 3:],
                    'pred': data[-len(data) // 3:-DTD]+list(pred.astype(int))})

    # Строим график с помощью plotly.express
    fig_ = px.line(df1, x='x', y=['test_data'], title='График исторических данных и предсказан')
    fig_.add_scatter(x=df1['x'].iloc[-DTD:], y=df1['pred'].iloc[-DTD:], mode='lines', name='pred')
    fig_.update_layout(xaxis=dict(autorange='reversed'))
    fig_.update_xaxes(tickvals=[200, 150, 100, 50, -1], 
                        ticktext=['200', '150', '100', '50', '-1'])
    # Устанавливаем диапазон по умолчанию для оси x (приближение)
    #fig_.update_xaxes(range=[5, 10])

    # Включаем rangeslider для возможности отдаления графика
    #fig_.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_)
    if len(pred) > 2:
        our_ns = pred[-3] - pred[-2]
        if our_ns < 0: our_ns = 0
        st.write(f'Прогноз неявок: {int(our_ns // 2)}')
