
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px


# Streamlit - allows ability to refresh app to see code changes
st.cache(allow_output_mutation=True)


st.title('Endowment Spending Simulator')

# Questions to gather input
cpi = st.sidebar.text_input("Enter an estimate for inflation (ex - difference between real and nominal expected return)")
t_intervals = st.sidebar.text_input("What is the simulation period? Enter in quarters (ex - 10 years = 40)")
options = st.sidebar.multiselect(
        'Select percentiles for comparison',
        [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
        )
rolling_quarters = st.sidebar.text_input("How many rolling quarters are used in the spending calculation methodlogy?")
uploaded_file = st.sidebar.file_uploader(f'Drop in Excel with {rolling_quarters} historical quarterly market values')
if uploaded_file is not None:
    historic_values = pd.read_excel(uploaded_file)


sim = []

# Functions to gather input for variables in each simulation

def annual_return():
    annual_ret = st.text_input(f"Nominal annualized expected return for Sim {sim}")
    return annual_ret

def annual_std():
    annual_std = st.text_input(f"Standard deviation for Sim {sim}")
    return annual_std

def annual_spending():
    annual_spending = st.text_input(f"Annual spend rate for Sim {sim}")
    return annual_spending

def annual_spending_initial():
    annual_spending_initial = st.text_input(f"First annual spend rate for Sim {sim}?")
    return annual_spending_initial

def annual_spending_initial_duration():
    annual_spending_initial_duration = st.text_input(f'First spend rate duration (quarters) for Sim {sim}?')
    return annual_spending_initial_duration

def annual_spending_final():
    annual_spending_final = st.text_input(f"Long-term annual spend rate for Sim {sim}?")
    return annual_spending_final

# Function to compute results assuming one constant spending rate

def compute_constant(annual_ret, annual_std, annual_spending, rolling_quarters, cpi, t_intervals):
            
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

    for t in range (rolling_quarters, t_intervals):
        IC_mv = pd.DataFrame(portfolio)
        IC_rolling_mv = IC_mv.rolling(rolling_quarters, min_periods=1).mean()
        IC_rolling_mv = np.array(IC_rolling_mv)
        spend[0] = quarterly_spending*IC_rolling_mv[0]
        spend[t] = quarterly_spending*IC_rolling_mv[t-1]
        portfolio[t] = (portfolio[t-1]*quarterly_returns[t])-spend[t]
        inflation_discounter[t] = inflation_discounter[t-1] * quarter_cpi[t]
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
    percentiles_real_spend = spend_real_df.quantile(options, axis = 1) 
    percentiles_real_spend = pd.DataFrame.transpose(percentiles_real_spend)



    percentiles_nominal = portfolio_nom_df.quantile(options, axis = 1)
    percentiles_nominal = pd.DataFrame.transpose(percentiles_nominal)
    percentiles_nom_spend = spend_nom_df.quantile(options, axis = 1) 
    percentiles_nom_spend = pd.DataFrame.transpose(percentiles_nom_spend)


    return portfolio_real, spend_real_df, portfolio, spend_nom_df, percentiles_real, percentiles_real_spend, percentiles_nominal, percentiles_nom_spend


# Function to compute results assuming a temporary short-term and long-term spending rate

def compute_variable(annual_ret, annual_std, annual_spending_initial, annual_spending_initial_duration, annual_spending_final, rolling_quarters, cpi, t_intervals):

    annual_ret = float(annual_ret) / 100
    annual_std = float(annual_std) / 100
    annual_spending_initial = float(annual_spending_initial) / 100
    annual_spending_final = float(annual_spending_final) / 100
    rolling_quarters = int(rolling_quarters)
    annual_spending_initial_duration = int(annual_spending_initial_duration) + rolling_quarters
    cpi = float(cpi) / 100
    t_intervals = int(t_intervals) + rolling_quarters

    quarterly_ret = annual_ret/4
    quarterly_stdev = annual_std / (4**0.5)
    quarterly_spending_initial = annual_spending_initial/4
    quarterly_spending_final = annual_spending_final/4
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
        if t <= annual_spending_initial_duration:
            quarterly_spending = quarterly_spending_initial
        else: 
            quarterly_spending = quarterly_spending_final
        spend[t] = quarterly_spending*IC_rolling_mv[t-1]
        portfolio[t] = (portfolio[t-1]*quarterly_returns[t])-spend[t]
        inflation_discounter[t] = inflation_discounter[t-1] * quarter_cpi[t]
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
    percentiles_real_spend = spend_real_df.quantile(options, axis = 1) 
    percentiles_real_spend = pd.DataFrame.transpose(percentiles_real_spend)


    percentiles_nominal = portfolio_nom_df.quantile(options, axis = 1)
    percentiles_nominal = pd.DataFrame.transpose(percentiles_nominal)
    percentiles_nom_spend = spend_nom_df.quantile(options, axis = 1) 
    percentiles_nom_spend = pd.DataFrame.transpose(percentiles_nom_spend)

    return portfolio_real, spend_real_df, portfolio, spend_nom_df, percentiles_real, percentiles_real_spend, percentiles_nominal, percentiles_nom_spend


# Streamlit browser display in columns

col1, col2, col3 = st.columns(3)
with col1:
    st.header("Sim 1")
    sim = 1
    spending_plan_1 = st.selectbox(
        '''Select a spending 
        plan for Sim 1''',
        ('', 'Constant (single spend rate)', 'Variable (multiple spend rates)')
    )
    if spending_plan_1 == 'Constant (single spend rate)':
        annual_return_1 = annual_return()
        annual_std_1 = annual_std()
        annual_spending_1 = annual_spending()
        
    elif spending_plan_1 == 'Variable (multiple spend rates)':
        annual_return_1 = annual_return()
        annual_std_1 = annual_std()
        annual_spending_initial_1 = annual_spending_initial()
        annual_spending_initial_duration_1 = annual_spending_initial_duration()
        annual_spending_final_1 = annual_spending_final()
        
with col2:
    st.header("Sim 2")
    sim = 2
    spending_plan_2 = st.selectbox(
        '''Select a spending 
        plan for Sim 2''',
        ('', 'Constant (single spend rate)', 'Variable (multiple spend rates)')
    )
    if spending_plan_2 == 'Constant (single spend rate)':
        annual_return_2 = annual_return()
        annual_std_2 = annual_std()
        annual_spending_2 = annual_spending()
    elif spending_plan_2 == 'Variable (multiple spend rates)':
        annual_return_2 = annual_return()
        annual_std_2 = annual_std()
        annual_spending_initial_2 = annual_spending_initial()
        annual_spending_initial_duration_2 = annual_spending_initial_duration()
        annual_spending_final_2 = annual_spending_final()

with col3:
    st.header("Sim 3")
    sim = 3
    spending_plan_3 = st.selectbox(
        '''Select a spending 
        plan for Sim 3''',
        ('', 'Constant (single spend rate)', 'Variable (multiple spend rates)')
    )
    if spending_plan_3 == 'Constant (single spend rate)':
        annual_return_3 = annual_return()
        annual_std_3 = annual_std()
        annual_spending_3 = annual_spending()
    elif spending_plan_3 == 'Variable (multiple spend rates)':
        annual_return_3 = annual_return()
        annual_std_3 = annual_std()
        annual_spending_initial_3 = annual_spending_initial()
        annual_spending_initial_duration_3 = annual_spending_initial_duration()
        annual_spending_final_3 = annual_spending_final()

# Select box for real or nominal terms
st.write('Select for Nominal Terms. Unselect for Real Terms.')
nom_check = st.checkbox('Nominal Terms')

# Computes the user input results if real terms is preferred
# Returns several dataframe/list objects to be used in charting below

if not nom_check and st.button('Compute'):
    if spending_plan_1 == 'Constant (single spend rate)':
        portfolio_real_df_1, spend_real_df_1, portfolio_nom_df_1, spend_nom_df_1, percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1 = compute_constant(annual_return_1, annual_std_1, annual_spending_1, rolling_quarters, cpi, t_intervals)
        
    elif spending_plan_1 == 'Variable (multiple spend rates)':
        portfolio_real_df_1, spend_real_df_1, portfolio_nom_df_1, spend_nom_df_1, percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1 = compute_variable(annual_return_1, annual_std_1, annual_spending_initial_1, annual_spending_initial_duration_1, annual_spending_final_1, rolling_quarters, cpi, t_intervals)
        
    if spending_plan_2 == 'Constant (single spend rate)':
        portfolio_real_df_2, spend_real_df_2, portfolio_nom_df_2, spend_nom_df_2, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_2, percentiles_nom_spend_2 = compute_constant(annual_return_2, annual_std_2, annual_spending_2, rolling_quarters, cpi, t_intervals)
        
    elif spending_plan_2 == 'Variable (multiple spend rates)':
        portfolio_real_df_2, spend_real_df_2, portfolio_nom_df_2, spend_nom_df_2, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_2, percentiles_nom_spend_2 = compute_variable(annual_return_2, annual_std_2, annual_spending_initial_2, annual_spending_initial_duration_2, annual_spending_final_2, rolling_quarters, cpi, t_intervals)

    if spending_plan_3 == 'Constant (single spend rate)':
        portfolio_real_df_3, spend_real_df_3, portfolio_nom_df_3, spend_nom_df_3, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_3, percentiles_nom_spend_3 = compute_constant(annual_return_3, annual_std_3, annual_spending_3, rolling_quarters, cpi, t_intervals)
        
    elif spending_plan_3 == 'Variable (multiple spend rates)':
        portfolio_real_df_3, spend_real_df_3, portfolio_nom_df_3, spend_nom_df_3, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_3, percentiles_nom_spend_3 = compute_variable(annual_return_3, annual_std_3, annual_spending_initial_3, annual_spending_initial_duration_3, annual_spending_final_3, rolling_quarters, cpi, t_intervals)



    # Takes the real portfolio market value results and organizes the output into a dataframe for the final ending market value only
    # if statements to allow flexibility if only 1 or 2 simulations conducted

    s1_mv = pd.DataFrame(portfolio_real_df_1[-1], columns=['Sim1'])
    if spending_plan_2 is not '':
        s2_mv = pd.DataFrame(portfolio_real_df_2[-1], columns=['Sim2'])
    if spending_plan_3 is not '':
        s3_mv = pd.DataFrame(portfolio_real_df_3[-1], columns=['Sim3'])

    # Combines each simulation into 1 dataframe
    if spending_plan_2 is '':
        s_combined_mv = s1_mv
    elif spending_plan_3 is '':
        s_combined_mv = pd.concat([s1_mv,s2_mv], axis=1)
    else:
        s_combined_mv = pd.concat([s1_mv,s2_mv,s3_mv], axis=1)

    # Repeats the above process for the $ spending data
    # Code is slightly different than above given the output of the spending data being in a different format
    s1_s = spend_real_df_1.cumsum().iloc[-1:].transpose()
    s1_s.rename(columns={s1_s.columns[0]: "Sim1" }, inplace = True)
   
    if spending_plan_2 is not '':
        s2_s = spend_real_df_2.cumsum().iloc[-1:].transpose()
        s2_s.rename(columns={s2_s.columns[0]: "Sim2" }, inplace = True)
    if spending_plan_3 is not '':
        s3_s = spend_real_df_3.cumsum().iloc[-1:].transpose()
        s3_s.rename(columns={s3_s.columns[0]: "Sim3" }, inplace = True)

    if spending_plan_2 is '':
        s_combined_s = s1_s
    elif spending_plan_3 is '':
        s_combined_s = pd.concat([s1_s,s2_s], axis=1)
    else:
        s_combined_s = pd.concat([s1_s,s2_s,s3_s], axis=1)

    # Takes the combined dataframe of ending markets values and reorganizes columns into one column
    s_combined_mv_stacked = s_combined_mv.stack().reset_index(level=0, drop=True)
    s_combined_mv_stacked = pd.DataFrame(s_combined_mv_stacked, columns = ['Ending Value'])
    s_combined_mv_stacked = s_combined_mv_stacked.reset_index()

    #  Takes the combined dataframe of $ spending values and reorganizes columns into one column
    s_combined_s_stacked = s_combined_s.stack().reset_index(drop=True)
    s_combined_s_stacked = pd.DataFrame(s_combined_s_stacked, columns = ['Total Dollars Spent'])

    # combines the two above
    s_combined_stacked_both = pd.concat([s_combined_mv_stacked, s_combined_s_stacked], axis=1)

    # Creates a scatter plot for ending MV and $ spent
    fig = px.scatter(s_combined_stacked_both, x="Ending Value", y="Total Dollars Spent", color='index', marginal_x='rug', marginal_y='rug', width=1200,height=1000, template='plotly_white',
            labels={"index": "Legend"}, opacity=0.3, trendline="ols")
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="right",x=0.89, font=dict(size= 20)))
    st.markdown('## Ending Portfolio Value vs. Total Dollars Spent')
    st.plotly_chart(fig)


    # Organizes dataframes for user selected percentiles
    for col in percentiles_real_1.columns[0:]:
        percentiles_real_1 = percentiles_real_1.rename(columns={col:'Sim1 MV '+ str(col)})

    if spending_plan_2 is not '':
        for col in percentiles_real_2.columns[0:]:
            percentiles_real_2 = percentiles_real_2.rename(columns={col:'Sim2 MV '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_real_3.columns[0:]:
            percentiles_real_3 = percentiles_real_3.rename(columns={col:'Sim3 MV '+ str(col)})


    # Plots user selected percentiles by MV

    fig = plt.figure(figsize=(10,6))
    plt.title('Market Value by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Portfolio Value')
    plt.plot(percentiles_real_1, color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_real_2, color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_real_3, color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    st.pyplot(fig)


    # Repeat for spending percentiles
    for col in percentiles_real_spend_1.columns[0:]:
        percentiles_real_spend_1 = percentiles_real_spend_1.rename(columns={col:'Sim1 Spend '+ str(col)})
    if spending_plan_2 is not '':
        for col in percentiles_real_spend_2.columns[0:]:
            percentiles_real_spend_2 = percentiles_real_spend_2.rename(columns={col:'Sim2 Spend '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_real_spend_3.columns[0:]:
            percentiles_real_spend_3 = percentiles_real_spend_3.rename(columns={col:'Sim3 Spend '+ str(col)})


    # Plots user selected percentiles by dollars spent

    fig = plt.figure(figsize=(10,6))
    plt.title('Quarterly Spending Power by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Portfolio Value')
    plt.plot(percentiles_real_spend_1, color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_real_spend_2, color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_real_spend_3, color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    st.pyplot(fig)


    # Plots user selected percentiles by cumulative dollars spent

    fig = plt.figure(figsize=(10,6))
    plt.title('Cumulative Dollars Spent by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Portfolio Value')
    plt.plot(percentiles_real_spend_1.cumsum(), color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_real_spend_2.cumsum(), color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_real_spend_3.cumsum(), color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    st.pyplot(fig)

    # Organizes percentiles into one combined dataframe to be able to download as CSV
    if spending_plan_2 is '':
        output = pd.concat([percentiles_real_1, percentiles_real_spend_1], axis=1)
    elif spending_plan_3 is '':
        output = pd.concat([percentiles_real_1, percentiles_real_spend_1, percentiles_real_2, percentiles_real_spend_2], axis=1)
    else:
        output = pd.concat([percentiles_real_1, percentiles_real_spend_1, percentiles_real_2, percentiles_real_spend_2, percentiles_real_3, percentiles_real_spend_3], axis=1)

    # Function to download output as CSV
    @st.cache
    def convert_df(df):
    #IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv_output = convert_df(output)

    st.write("Click to download results as a CSV")
    st.download_button(
        label="Download",
        data=csv_output,
        file_name='output.csv',
        mime='text/csv',
    )



# Computes the user input results if real terms is preferred
# Returns several dataframe/list objects to be used in charting below

elif nom_check and st.button('Compute'):
    if spending_plan_1 == 'Constant (single spend rate)':
        portfolio_real_df_1, spend_real_df_1, portfolio_nom_df_1, spend_nom_df_1, percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1 = compute_constant(annual_return_1, annual_std_1, annual_spending_1, rolling_quarters, cpi, t_intervals)
        
    elif spending_plan_1 == 'Variable (multiple spend rates)':
        portfolio_real_df_1, spend_real_df_1, portfolio_nom_df_1, spend_nom_df_1, percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1 = compute_variable(annual_return_1, annual_std_1, annual_spending_initial_1, annual_spending_initial_duration_1, annual_spending_final_1, rolling_quarters, cpi, t_intervals)
        
    if spending_plan_2 == 'Constant (single spend rate)':
        portfolio_real_df_2, spend_real_df_2, portfolio_nom_df_2, spend_nom_df_2, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_2, percentiles_nom_spend_2 = compute_constant(annual_return_2, annual_std_2, annual_spending_2, rolling_quarters, cpi, t_intervals)
        
    elif spending_plan_2 == 'Variable (multiple spend rates)':
        portfolio_real_df_2, spend_real_df_2, portfolio_nom_df_2, spend_nom_df_2, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_2, percentiles_nom_spend_2 = compute_variable(annual_return_2, annual_std_2, annual_spending_initial_2, annual_spending_initial_duration_2, annual_spending_final_2, rolling_quarters, cpi, t_intervals)

    if spending_plan_3 == 'Constant (single spend rate)':
        portfolio_real_df_3, spend_real_df_3, portfolio_nom_df_3, spend_nom_df_3, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_3, percentiles_nom_spend_3 = compute_constant(annual_return_3, annual_std_3, annual_spending_3, rolling_quarters, cpi, t_intervals)
        
    elif spending_plan_3 == 'Variable (multiple spend rates)':
        portfolio_real_df_3, spend_real_df_3, portfolio_nom_df_3, spend_nom_df_3, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_3, percentiles_nom_spend_3 = compute_variable(annual_return_3, annual_std_3, annual_spending_initial_3, annual_spending_initial_duration_3, annual_spending_final_3, rolling_quarters, cpi, t_intervals)



    # Takes the real portfolio market value results and organizes the output into a dataframe for the final ending market value only
    # if statements to allow flexibility if only 1 or 2 simulations conducted

    s1_mv = pd.DataFrame(portfolio_nom_df_1[-1], columns=['Sim1'])
    if spending_plan_2 is not '':
        s2_mv = pd.DataFrame(portfolio_nom_df_2[-1], columns=['Sim2'])
    if spending_plan_3 is not '':
        s3_mv = pd.DataFrame(portfolio_nom_df_3[-1], columns=['Sim3'])

    # Combines each simulation into 1 dataframe
    if spending_plan_2 is '':
        s_combined_mv = s1_mv
    elif spending_plan_3 is '':
        s_combined_mv = pd.concat([s1_mv,s2_mv], axis=1)
    else:
        s_combined_mv = pd.concat([s1_mv,s2_mv,s3_mv], axis=1)

    # Repeats the above process for the $ spending data
    # Code is slightly different than above given the output of the spending data being in a different format

    s1_s = spend_nom_df_1.cumsum().iloc[-1:].transpose()
    s1_s.rename(columns={s1_s.columns[0]: "Sim1" }, inplace = True)
   
    if spending_plan_2 is not '':
        s2_s = spend_nom_df_2.cumsum().iloc[-1:].transpose()
        s2_s.rename(columns={s2_s.columns[0]: "Sim2" }, inplace = True)
    if spending_plan_3 is not '':
        s3_s = spend_nom_df_3.cumsum().iloc[-1:].transpose()
        s3_s.rename(columns={s3_s.columns[0]: "Sim3" }, inplace = True)

    if spending_plan_2 is '':
        s_combined_s = s1_s
    elif spending_plan_3 is '':
        s_combined_s = pd.concat([s1_s,s2_s], axis=1)
    else:
        s_combined_s = pd.concat([s1_s,s2_s,s3_s], axis=1)

    # Takes the combined dataframe of ending markets values and reorganizes columns into one column
    s_combined_mv_stacked = s_combined_mv.stack().reset_index(level=0, drop=True)
    s_combined_mv_stacked = pd.DataFrame(s_combined_mv_stacked, columns = ['Ending Value'])
    s_combined_mv_stacked = s_combined_mv_stacked.reset_index()

    #  Takes the combined dataframe of $ spending values and reorganizes columns into one column
    s_combined_s_stacked = s_combined_s.stack().reset_index(drop=True)
    s_combined_s_stacked = pd.DataFrame(s_combined_s_stacked, columns = ['Total Dollars Spent'])

    # combines the two above
    s_combined_stacked_both = pd.concat([s_combined_mv_stacked, s_combined_s_stacked], axis=1)

    # Creates a scatter plot for ending MV and $ spent
    fig = px.scatter(s_combined_stacked_both, x="Ending Value", y="Total Dollars Spent", color='index', marginal_x='rug', marginal_y='rug', width=1200,height=1000, template='plotly_white',
            labels={"index": "Legend"}, opacity=0.3, trendline="ols")
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="right",x=0.89, font=dict(size= 20)))
    st.markdown('## Ending Portfolio Value vs. Total Dollars Spent')
    st.plotly_chart(fig)

    # Organizes dataframes for user selected percentiles

    for col in percentiles_nominal_1.columns[0:]:
        percentiles_nominal_1 = percentiles_nominal_1.rename(columns={col:'Sim1 MV '+ str(col)})
    if spending_plan_2 is not '':
        for col in percentiles_nominal_2.columns[0:]:
            percentiles_nominal_2 = percentiles_nominal_2.rename(columns={col:'Sim2 MV '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_nominal_3.columns[0:]:
            percentiles_nominal_3 = percentiles_nominal_3.rename(columns={col:'Sim3 MV '+ str(col)})


    # Plots user selected percentiles by MV
    fig = plt.figure(figsize=(10,6))
    plt.title('Market Value by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Portfolio Value')
    plt.plot(percentiles_nominal_1, color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_nominal_2, color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_nominal_3, color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    st.pyplot(fig)


    # Repeat for spending percentiles
    for col in percentiles_nom_spend_1.columns[0:]:
        percentiles_nom_spend_1 = percentiles_nom_spend_1.rename(columns={col:'Sim1 Spend '+ str(col)})
    if spending_plan_2 is not '':
        for col in percentiles_nom_spend_2.columns[0:]:
            percentiles_nom_spend_2 = percentiles_nom_spend_2.rename(columns={col:'Sim2 Spend '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_nom_spend_3.columns[0:]:
            percentiles_nom_spend_3 = percentiles_nom_spend_3.rename(columns={col:'Sim3 Spend '+ str(col)})

    # Plots user selected percentiles by $ spent
    fig = plt.figure(figsize=(10,6))
    plt.title('Quarterly Spending Power by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Portfolio Value')
    plt.plot(percentiles_nom_spend_1, color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_nom_spend_2, color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_nom_spend_3, color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    st.pyplot(fig)

    # Plots user selected percentiles by cumulative $ spent
    fig = plt.figure(figsize=(10,6))
    plt.title('Cumulative Dollars Spent by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Portfolio Value')
    plt.plot(percentiles_nom_spend_1.cumsum(), color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_nom_spend_2.cumsum(), color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_nom_spend_3.cumsum(), color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    st.pyplot(fig)

    # Organizes percentiles into one combined dataframe to be able to download as CSV
    if spending_plan_2 is '':
        output = pd.concat([percentiles_nominal_1, percentiles_nom_spend_1], axis=1)
    elif spending_plan_3 is '':
        output = pd.concat([percentiles_nominal_1, percentiles_nom_spend_1, percentiles_nominal_2, percentiles_nom_spend_2], axis=1)
    else:
        output = pd.concat([percentiles_nominal_1, percentiles_nom_spend_1, percentiles_nominal_2, percentiles_nom_spend_2, percentiles_nominal_3, percentiles_nom_spend_3], axis=1)

    # Function to download output as CSV
    @st.cache
    def convert_df(df):
    #IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv_output = convert_df(output)

    st.write("Click to download results as a CSV")
    st.download_button(
        label="Download",
        data=csv_output,
        file_name='output.csv',
        mime='text/csv',
    )
