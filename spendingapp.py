import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import norm
#import math
#import random
#import statistics
import streamlit as st
#import time



# Streamlit - allows ability to refresh app to see code changes
st.cache(allow_output_mutation=True)


st.title('Endowment Spending Simulator')

# Questions to gather input
spending_plan = st.sidebar.selectbox(
    'What type of spending plan?',
    ('Constant (single spend rate)', 'Variable (multiple spend rates)')
)

if spending_plan == 'Constant (single spend rate)':

    annual_ret = st.sidebar.text_input("Enter the nominal annualized expected return")
    annual_std = st.sidebar.text_input("Enter the expected standard deviation")
    annual_spending = st.sidebar.text_input("What is the annual spend rate?")
    rolling_quarters = st.sidebar.text_input("Enter the rolling period (quarters) for spending formula")
    cpi = st.sidebar.text_input("Enter an estimate for inflation (ex - difference between real and nominal expected return)")
    t_intervals = st.sidebar.text_input("What is the simulation period? Enter in quarters (ex - 10 years = 40)")
    options = st.sidebar.multiselect(
        'Select percentiles for comparison',
        [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
        )
    uploaded_file = st.sidebar.file_uploader('Drop in Excel with Historical Market Values')
    if uploaded_file is not None:
        historic_values = pd.read_excel(uploaded_file)

    if st.sidebar.button('Compute'):
        
        annual_ret = float(annual_ret) / 100
        annual_std = float(annual_std) / 100
        annual_spending = float(annual_spending) / 100
        rolling_quarters = int(rolling_quarters)
        cpi = float(cpi) / 100
        t_intervals = int(t_intervals) + rolling_quarters

        quarterly_ret = annual_ret/4
        quarterly_spending = annual_spending/4
        quarterly_stdev = annual_std / (4**0.5)
        quarterly_returns = 1 + np.random.normal(quarterly_ret, quarterly_stdev, (t_intervals,10001))
        spend = np.zeros_like(quarterly_returns)
        portfolio = np.zeros_like(quarterly_returns)
        portfolio[0:rolling_quarters]= historic_values[0:rolling_quarters]
        quarter_cpi = cpi / 4
        quarter_cpi = 1 + np.random.normal(quarter_cpi, .009, (t_intervals,10001))
        portfolio_real = np.zeros_like(quarterly_returns)
        portfolio_real[0:rolling_quarters] = historic_values[0:rolling_quarters]
        spend_real = np.zeros_like(quarterly_returns)
        spend_real[0:rolling_quarters] = 0
        time_zero = rolling_quarters - 1
        inflation_discounter = np.zeros_like(quarterly_returns)
        inflation_discounter[0:rolling_quarters] = 1
        
        #simulation
        for t in range (rolling_quarters, t_intervals):
            IC_mv = pd.DataFrame(portfolio)
            IC_rolling_mv = IC_mv.rolling(rolling_quarters, min_periods=1).mean()
            IC_rolling_mv = np.array(IC_rolling_mv)
            spend[0] = quarterly_spending*IC_rolling_mv[0]
            spend[t] = quarterly_spending*IC_rolling_mv[t-1]
            portfolio[t] = (portfolio[t-1]*quarterly_returns[t])-spend[t]
            inflation_discounter[t] = inflation_discounter[t-1] * quarter_cpi[t]
            #IC_mv_real = pd.DataFrame(portfolio_real)
            #IC_spend_real = pd.DataFrame(spend_real)
            portfolio_real[t] = (portfolio[t] / inflation_discounter[t])
            spend_real[t] = (spend[t] / inflation_discounter[t])

        portfolio_real_df =  pd.DataFrame(portfolio_real[time_zero:]).reset_index(drop=True)
        spend_real_df = pd.DataFrame(spend_real[rolling_quarters:]).reset_index(drop=True)
        spend_real_df.index = np.arange(1, len(spend_real_df)+1)

        portfolio_nom_df =  pd.DataFrame(portfolio[time_zero:]).reset_index(drop=True)
        spend_nom_df = pd.DataFrame(spend[rolling_quarters:]).reset_index(drop=True)
        spend_nom_df.index = np.arange(1, len(spend_nom_df)+1)


        percentiles_real = portfolio_real_df.quantile(options, axis = 1)
        percentiles_real = pd.DataFrame.transpose(percentiles_real)
        st.markdown('## Projected Real Market Values')
        st.write(f'Expected real return of {((annual_ret - cpi)*100):.2f}% (annual return less CPI)')
        st.line_chart(percentiles_real)
        percentiles_real_spend = spend_real_df.quantile(options, axis = 1) 
        percentiles_real_spend = pd.DataFrame.transpose(percentiles_real_spend)
        st.markdown('## Projected Real Dollars Spent')
        st.line_chart(percentiles_real_spend)
        percentiles_real_combined = pd.concat([percentiles_real,percentiles_real_spend], axis=1)


        percentiles_nominal = portfolio_nom_df.quantile(options, axis = 1)
        percentiles_nominal = pd.DataFrame.transpose(percentiles_nominal)
        st.markdown('## Projected Nominal Market Values')
        st.write(f'Expected nominal return of {(annual_ret*100):.2f}%')
        st.line_chart(percentiles_nominal)
        percentiles_nom_spend = spend_nom_df.quantile(options, axis = 1) 
        percentiles_nom_spend = pd.DataFrame.transpose(percentiles_nom_spend)
        st.markdown('## Projected Nominal Dollars Spent')
        st.line_chart(percentiles_nom_spend)
        percentiles_nom_combined = pd.concat([percentiles_nominal,percentiles_nom_spend], axis=1)


        output = pd.concat([percentiles_real_combined,percentiles_nom_combined], axis=1)

        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv_output = convert_df(output)

        st.write("Click the button below to download the output as a CSV")
        st.write("CSV columns will appear in the following order: Real Market Values, Real Dollars Spent, Nominal Market Values, Nominal Dollars Spent")
        st.download_button(
            label="Download",
            data=csv_output,
            file_name='output.csv',
            mime='text/csv',
        )

elif spending_plan == 'Variable (multiple spend rates)':

    annual_ret = st.sidebar.text_input("Enter the nominal annualized expected return")
    annual_std = st.sidebar.text_input("Enter the expected standard deviation")
    annual_spending_1 = st.sidebar.text_input("What is the first annual spend rate?")
    spending_1_duration = st.sidebar.text_input('How long (in quarters) will the initial spend rate be implemented?')
    annual_spending_2 = st.sidebar.text_input("What is the long-term (second) annual spend rate?")
    rolling_quarters = st.sidebar.text_input("Enter the rolling period (quarters) for spending formula")
    cpi = st.sidebar.text_input("Enter an estimate for inflation (ex - difference between real and nominal expected return)")
    t_intervals = st.sidebar.text_input("What is the simulation period? Enter in quarters (ex - 10 years = 40)")
    options = st.sidebar.multiselect(
        'Select percentiles for comparison',
        [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
        )
    uploaded_file = st.sidebar.file_uploader('Drop in Excel with Historical Market Values')
    if uploaded_file is not None:
        historic_values = pd.read_excel(uploaded_file)

    
    if st.sidebar.button('Compute'):
        
        annual_ret = float(annual_ret) / 100
        annual_std = float(annual_std) / 100
        annual_spending_1 = float(annual_spending_1) / 100
        annual_spending_2 = float(annual_spending_2) / 100
        rolling_quarters = int(rolling_quarters)
        spending_1_duration = int(spending_1_duration) + rolling_quarters
        cpi = float(cpi) / 100
        t_intervals = int(t_intervals) + rolling_quarters

        quarterly_ret = annual_ret/4
        quarterly_stdev = annual_std / (4**0.5)
        quarterly_spending_1 = annual_spending_1/4
        quarterly_spending_2 = annual_spending_2/4
        quarterly_returns = 1 + np.random.normal(quarterly_ret, quarterly_stdev, (t_intervals,10001))
        spend = np.zeros_like(quarterly_returns)
        portfolio = np.zeros_like(quarterly_returns)
        portfolio[0:rolling_quarters]= historic_values[0:rolling_quarters]
        quarter_cpi = cpi / 4
        quarter_cpi = 1 + np.random.normal(quarter_cpi, .009, (t_intervals,10001))
        portfolio_real = np.zeros_like(quarterly_returns)
        portfolio_real[0:rolling_quarters] = historic_values[0:rolling_quarters]
        spend_real = np.zeros_like(quarterly_returns)
        spend_real[0:rolling_quarters] = 0
        time_zero = rolling_quarters - 1
        inflation_discounter = np.zeros_like(quarterly_returns)
        inflation_discounter[0:rolling_quarters] = 1
        
        #simulation
        for t in range (rolling_quarters, t_intervals):
            IC_mv = pd.DataFrame(portfolio)
            IC_rolling_mv = IC_mv.rolling(rolling_quarters, min_periods=1).mean()
            IC_rolling_mv = np.array(IC_rolling_mv)
            if t <= spending_1_duration:
                quarterly_spending = quarterly_spending_1
            else: 
                quarterly_spending = quarterly_spending_2
            spend[t] = quarterly_spending*IC_rolling_mv[t-1]
            portfolio[t] = (portfolio[t-1]*quarterly_returns[t])-spend[t]
            inflation_discounter[t] = inflation_discounter[t-1] * quarter_cpi[t]
            #IC_mv_real = pd.DataFrame(portfolio_real)
            #IC_spend_real = pd.DataFrame(spend_real)
            portfolio_real[t] = (portfolio[t] / inflation_discounter[t])
            spend_real[t] = (spend[t] / inflation_discounter[t])


        portfolio_real_df =  pd.DataFrame(portfolio_real[time_zero:]).reset_index(drop=True)
        spend_real_df = pd.DataFrame(spend_real[rolling_quarters:]).reset_index(drop=True)
        spend_real_df.index = np.arange(1, len(spend_real_df)+1)

        portfolio_nom_df =  pd.DataFrame(portfolio[time_zero:]).reset_index(drop=True)
        spend_nom_df = pd.DataFrame(spend[rolling_quarters:]).reset_index(drop=True)
        spend_nom_df.index = np.arange(1, len(spend_nom_df)+1)


        percentiles_real = portfolio_real_df.quantile(options, axis = 1)
        percentiles_real = pd.DataFrame.transpose(percentiles_real)
        st.markdown('## Projected Real Market Values')
        st.write(f'Expected real return of {((annual_ret - cpi)*100):.2f}% (annual return less CPI)')
        st.line_chart(percentiles_real)
        percentiles_real_spend = spend_real_df.quantile(options, axis = 1) 
        percentiles_real_spend = pd.DataFrame.transpose(percentiles_real_spend)
        st.markdown('## Projected Real Dollars Spent')
        st.line_chart(percentiles_real_spend)
        percentiles_real_combined = pd.concat([percentiles_real,percentiles_real_spend], axis=1)


        percentiles_nominal = portfolio_nom_df.quantile(options, axis = 1)
        percentiles_nominal = pd.DataFrame.transpose(percentiles_nominal)
        st.markdown('## Projected Nominal Market Values')
        st.write(f'Expected nominal return of {annual_ret*100}%')
        st.line_chart(percentiles_nominal)
        percentiles_nom_spend = spend_nom_df.quantile(options, axis = 1) 
        percentiles_nom_spend = pd.DataFrame.transpose(percentiles_nom_spend)
        st.markdown('## Projected Nominal Dollars Spent')
        st.line_chart(percentiles_nom_spend)
        percentiles_nom_combined = pd.concat([percentiles_nominal,percentiles_nom_spend], axis=1)


        output = pd.concat([percentiles_real_combined,percentiles_nom_combined], axis=1)

        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv_output = convert_df(output)

        st.write("Click the button below to download the output as a CSV")
        st.write("CSV columns will appear in the following order: Real Market Values, Real Dollars Spent, Nominal Market Values, Nominal Dollars Spent")
        st.download_button(
            label="Download",
            data=csv_output,
            file_name='output.csv',
            mime='text/csv',
        )