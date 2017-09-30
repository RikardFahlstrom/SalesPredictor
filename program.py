from workalendar import europe, africa, america, asia, usa
import pandas as pd
from fbprophet import Prophet
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 10)


def main():
    header()
    file_path = 'FILEPATH'
    raw_data = get_data_from_excel(file_path)
    sales_data = prepare_sales_data(raw_data)
    holidays = get_past_and_future_holidays()
    forecast = prediction_with_holidays(holidays, sales_data)
    final_df = plot_forecast(forecast, sales_data, raw_data)
    save_to_csv(final_df, 'FILENAME')


def header():
    print('-----------------------------------------')
    print('        PREDICTOR - INSTRUCTIONS')
    print('-----------------------------------------')
    print('1) Add filepath in main-function. Line 13')
    print('2) Input file expects to contain one column with dates and second column with data to predict')
    print('3) Add data for period 2016 - current')
    print("4) Change 'index_col' on line 36 to column name for date column, same for line 49")
    print("5) Change filename in 'save_to_csv'-function. Line 19 ")
    print('6) Change to relevant holiday country on lines 66-68')
    print('7) Your output (csv) will be save to working directory')
    print('-----------------------------------------')


def get_data_from_excel(file_path):
    sales_df = pd.read_excel(file_path, index_col='COLUMN FOR DATES', parse_dates=True)
    #  print('--------------------------------------')
    #  print(sales_df.dtypes)
    #  print('--------------------------------------')
    #  print('sales_df looks like: ')
    #  print(sales_df)

    return sales_df


def prepare_sales_data(raw_data):

    df = raw_data.reset_index()
    df['COLUMN FOR DATES'] = pd.to_datetime(df['COLUMN FOR DATES'])
    df.columns = ['ds', 'y']

    #  df.plot(x='ds', y='y')
    #  plt.show()
    #  plt.waitforbuttonpress()

    df['y'] = np.log(df['y'])

    #  print('--------------------------------------')
    #  print(df.dtypes)
    #  print('--------------------------------------')
    #  print('df dataframe looks like: ')
    #  print(df)

    return df


def get_past_and_future_holidays():

    country_2016 = europe.France().holidays(2016)
    country_2017 = europe.France().holidays(2017)
    country_2018 = europe.France().holidays(2018)

    labels = ['ds', 'holiday']

    df_holiday_2016 = pd.DataFrame.from_records(country_2016, columns=labels)
    df_holiday_2016['ds'] = pd.to_datetime(df_holiday_2016['ds'])

    df_holiday_2017 = pd.DataFrame.from_records(country_2017, columns=labels)
    df_holiday_2017['ds'] = pd.to_datetime(df_holiday_2017['ds'])

    df_holiday_2018 = pd.DataFrame.from_records(country_2018, columns=labels)
    df_holiday_2018['ds'] = pd.to_datetime(df_holiday_2018['ds'])

    df_holidays = df_holiday_2016.append([df_holiday_2017, df_holiday_2018], ignore_index=True)
    df_holidays['lower_window'] = 0
    df_holidays['upper_window'] = 0
    df_holidays['holiday'] = 'publish'
    df_holidays['holiday'] = df_holidays['holiday'].astype('str')

    #  print('-------------------------------------- ')
    #  print('df_holiday dataframe looks like: ')
    #  print(df_holidays.dtypes)
    #  print('-------------------------------------- ')
    #  print(df_holidays)

    return df_holidays


def prediction_with_holidays(holiday_df, df):

    model = Prophet(holidays=holiday_df)
    model.fit(df)
    future = model.make_future_dataframe(periods=90, freq='d')
    #  print('future tail dataframe')
    #  print(future.tail())

    forecast = model.predict(future)

    #  print('forecast tail dataframe')
    #  print(forecast.tail())

    #  print('forecast df:s relevant columns:')
    #  print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    #  model.plot(forecast)
    #  plt.show()

    return forecast


def plot_forecast(forecast, df, raw_data):
    df.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)
    
    viz_df = raw_data.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')

    #  print('viz_df looks like: ')
    #  print(viz_df.head())

    viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])

    #  print('viz_df looks like: ')
    #  print(viz_df.head())

    #  Compares our original data with the estimated figures
    #  viz_df[['NumOrders', 'yhat_rescaled']].plot()
    #  plt.show()

    raw_data.index = pd.to_datetime(raw_data.index)  # need to work with datetime objects
    connect_date = raw_data.index[-2]  # select 2nd to last date

    mask = (forecast.index > connect_date)
    predict_df = forecast.loc[mask]

    #  print('new predict_df looks like: ')
    #  print(predict_df.head())

    viz_df = raw_data.join(predict_df[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')
    viz_df['yhat_scaled'] = np.exp(viz_df['yhat'])

    print('viz_df looks like: ')
    print(viz_df)

    fig, ax1 = plt.subplots()
    ax1.plot(viz_df.NumOrders)
    ax1.plot(viz_df.yhat_scaled, color='black', linestyle=':')
    ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5,
                     color='darkgray')
    ax1.set_title('Actual Orders (Orange) vs Estimated Orders (Black) with uncertainty interval (Grey)')
    ax1.set_ylabel('NumOrders')
    ax1.set_xlabel('Date')

    l = ax1.legend()  # get the legend
    l.get_texts()[0].set_text('Actual Orders')  # change the legend text for 1st plot
    l.get_texts()[1].set_text('Estimated Orders')  # change the legend text for 2nd plot

    plt.show()

    return viz_df


def save_to_csv(df, name):
    df.to_csv('{}.csv'.format(name), encoding='utf-8')


if __name__ == '__main__':
    main()
