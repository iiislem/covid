import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import warnings
import io
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="COVID-19 Prediction Dashboard",
    page_icon="🦠",
    layout="wide"
)

# Set better plot style
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# App title and description
st.title("COVID-19 Analysis and Prediction Dashboard")
st.markdown("""
This application analyzes COVID-19 data using advanced hybrid modeling (SIR epidemiological model + Deep Learning) 
to predict future cases, deaths, and recoveries for selected countries with high accuracy.
""")

# Function to load country population data
@st.cache_data
def load_population_data():
    try:
        # Replace with your actual file path
        population_df = pd.read_csv("world_population.csv")
        # Ensure correct column names
        if 'Country/Territory' in population_df.columns and '2020 Population' in population_df.columns:
            # Create a dictionary for easy lookup
            population_dict = dict(zip(population_df['Country/Territory'], population_df['2020 Population']))
            return population_dict
        else:
            st.error("Population data CSV should contain 'Country/Territory' and '2020 Population' columns.")
            return {}
    except Exception as e:
        st.error(f"Error loading population data: {e}")
        return {}

# Function to generate mock COVID data for testing
@st.cache_data
def load_covid_data():
    try:
        # Use the path provided in your code
        df = pd.read_csv("covid_19_clean_complete.csv")
        
        # Data preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df['Province/State'].fillna('', inplace=True)
        df['Confirmed'].fillna(0, inplace=True)
        df['Deaths'].fillna(0, inplace=True)
        df['Recovered'].fillna(0, inplace=True)
        df['Active'].fillna(0, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading COVID-19 data: {e}")
        st.stop()

# Function to create matplotlib figure and convert to Streamlit with better styling
def create_styled_figure(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

# Function to format large numbers
def format_ticks(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.1f}K'
    else:
        return f'{x:.0f}'

# Initialize session state for plots
if 'sir_plot_data' not in st.session_state:
    st.session_state['sir_plot_data'] = None
if 'dl_plot_data' not in st.session_state:
    st.session_state['dl_plot_data'] = None

# Load COVID data
with st.spinner("Loading data..."):
    df = load_covid_data()
    population_dict = load_population_data()

# Sidebar for user inputs
st.sidebar.header("Settings")

# Get unique countries for dropdown
unique_countries = sorted(df['Country/Region'].unique())

# Country selection (ONLY in sidebar)
country = st.sidebar.selectbox("Select Country:", unique_countries)

# Get population for selected country automatically
if country in population_dict:
    population = population_dict[country]
    st.sidebar.success(f"Population loaded: {population:,}")
else:
    # Default population if not found
    population = st.sidebar.number_input("Population not found. Please enter manually:", 
                                        min_value=1000, 
                                        value=1000000, 
                                        step=1000)

# Date selection with start from January 2020
min_date = datetime.date(2020, 1, 1)
latest_date = df['Date'].max().date()
max_date = datetime.date.today()
default_date = min(latest_date + timedelta(days=30), max_date)

# Use date_input with explicit min and max values
prediction_date = st.sidebar.date_input(
    "Predict for date:", 
    value=default_date,
    min_value=min_date,
    max_value=max_date
)

# Custom SIR parameters
use_custom_sir = st.sidebar.checkbox("Use custom SIR values?")

if use_custom_sir:
    sir_col1, sir_col2, sir_col3 = st.sidebar.columns(3)
    with sir_col1:
        susceptible = st.number_input("Susceptible:", min_value=0, value=int(population * 0.9))
    with sir_col2:
        infected = st.number_input("Infected:", min_value=0, value=5000)
    with sir_col3:
        recovered = st.number_input("Recovered:", min_value=0, value=int(population * 0.1))
else:
    # These will be calculated from the data later
    susceptible, infected, recovered = None, None, None

# Prepare country-specific data
country_df = df[df['Country/Region'] == country].copy()

# Aggregate by date if multiple provinces
daily_data = country_df.groupby('Date').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum',
    'Active': 'sum'
}).reset_index()

# Sort by date
daily_data = daily_data.sort_values('Date')

# Apply rolling average to smooth data
window_size = 7  # 7-day rolling average
daily_data['Confirmed_Smooth'] = daily_data['Confirmed'].rolling(window=window_size, center=False).mean().fillna(daily_data['Confirmed'])
daily_data['Deaths_Smooth'] = daily_data['Deaths'].rolling(window=window_size, center=False).mean().fillna(daily_data['Deaths'])
daily_data['Recovered_Smooth'] = daily_data['Recovered'].rolling(window=window_size, center=False).mean().fillna(daily_data['Recovered'])
daily_data['Active_Smooth'] = daily_data['Active'].rolling(window=window_size, center=False).mean().fillna(daily_data['Active'])

# Calculate daily new cases
daily_data['New_Confirmed'] = daily_data['Confirmed'].diff().fillna(0).clip(lower=0)
daily_data['New_Deaths'] = daily_data['Deaths'].diff().fillna(0).clip(lower=0)
daily_data['New_Recovered'] = daily_data['Recovered'].diff().fillna(0).clip(lower=0)

# Apply rolling average to new cases
daily_data['New_Confirmed_Smooth'] = daily_data['New_Confirmed'].rolling(window=window_size, center=False).mean().fillna(daily_data['New_Confirmed'])
daily_data['New_Deaths_Smooth'] = daily_data['New_Deaths'].rolling(window=window_size, center=False).mean().fillna(daily_data['New_Deaths'])
daily_data['New_Recovered_Smooth'] = daily_data['New_Recovered'].rolling(window=window_size, center=False).mean().fillna(daily_data['New_Recovered'])

# Display current statistics
latest_stats = daily_data.iloc[-1]
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.metric("Total Confirmed", f"{int(latest_stats['Confirmed']):,}", f"+{int(latest_stats['New_Confirmed']):,}")
with stats_col2:
    st.metric("Total Deaths", f"{int(latest_stats['Deaths']):,}", f"+{int(latest_stats['New_Deaths']):,}")
with stats_col3:
    st.metric("Total Recovered", f"{int(latest_stats['Recovered']):,}", f"+{int(latest_stats['New_Recovered']):,}")
with stats_col4:
    st.metric("Active Cases", f"{int(latest_stats['Active']):,}")

# Main overview chart
st.subheader(f"COVID-19 Data for {country}")
fig1, ax1 = create_styled_figure(figsize=(12, 6))

ax1.plot(daily_data['Date'], daily_data['Confirmed_Smooth'], label='Confirmed', linewidth=2.5, color='#3366CC')
ax1.plot(daily_data['Date'], daily_data['Deaths_Smooth'], label='Deaths', linewidth=2.5, color='#DC3912')
ax1.plot(daily_data['Date'], daily_data['Recovered_Smooth'], label='Recovered', linewidth=2.5, color='#109618')
ax1.plot(daily_data['Date'], daily_data['Active_Smooth'], label='Active', linewidth=2.5, color='#FF9900')

# Format date axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Format y-axis for large numbers
ax1.yaxis.set_major_formatter(FuncFormatter(format_ticks))

ax1.set_title(f'COVID-19 Cumulative Cases for {country}', fontweight='bold')
ax1.set_xlabel('Date', fontweight='bold')
ax1.set_ylabel('Number of Cases', fontweight='bold')
ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
ax1.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# Enhanced SIR model with more realistic dynamics
def enhanced_sir_model(t, y, beta, gamma, mu, alpha, population):
    """Enhanced SIR model with births, deaths, and vaccination."""
    S, I, R, D = y
    N = S + I + R
    
    # Time-varying parameters
    beta_t = beta * (1 - alpha * np.tanh(t / 100))  # Decreasing transmission due to interventions
    gamma_t = gamma * (1 + 0.1 * np.tanh(t / 50))   # Improving recovery rate
    
    dSdt = mu * population - beta_t * S * I / N - mu * S
    dIdt = beta_t * S * I / N - gamma_t * I - mu * I
    dRdt = gamma_t * I - mu * R
    dDdt = mu * I * 0.02  # Death rate
    
    return [dSdt, dIdt, dRdt, dDdt]

# Function to run enhanced SIR model
def run_sir_model(daily_data, population, prediction_date):
    # Set up SIR model initial conditions
    I0 = max(1, daily_data['Active'].iloc[-1])
    R0_total = daily_data['Recovered'].iloc[-1]
    D0 = daily_data['Deaths'].iloc[-1]
    S0 = population - I0 - R0_total - D0
    
    initial_conditions = [S0, I0, R0_total, D0]
    
    # Time points
    t = np.arange(len(daily_data))
    
    # Function to minimize for SIR model
    def loss_function(params):
        beta, gamma, mu, alpha = params
        try:
            solution = solve_ivp(
                fun=lambda t, y: enhanced_sir_model(t, y, beta, gamma, mu, alpha, population),
                t_span=[0, max(t)],
                y0=initial_conditions,
                t_eval=t,
                method='RK45',
                rtol=1e-6
            )
            
            if solution.success:
                S, I, R, D = solution.y
                
                # Calculate cumulative cases
                C = I + R + D
                
                loss_I = np.mean((I - daily_data['Active_Smooth'].values)**2)
                loss_R = np.mean((R - daily_data['Recovered_Smooth'].values)**2)
                loss_D = np.mean((D - daily_data['Deaths_Smooth'].values)**2)
                loss_C = np.mean((C - daily_data['Confirmed_Smooth'].values)**2)
                
                return loss_I + loss_R + loss_D + 0.1 * loss_C
            else:
                return 1e10
        except:
            return 1e10
    
    # Optimization with realistic bounds
    bounds = [(0.01, 1.0), (0.01, 0.5), (0.0001, 0.01), (0.0, 0.9)]
    
    result = differential_evolution(
        loss_function, 
        bounds, 
        maxiter=100,
        seed=42
    )
    
    # Store optimized parameters
    beta_opt, gamma_opt, mu_opt, alpha_opt = result.x
    
    # Calculate predictions
    latest_date = daily_data['Date'].iloc[-1]
    prediction_days = (prediction_date - latest_date.to_pydatetime().date()).days
    
    if prediction_days <= 0:
        prediction_days = 30
    
    prediction_dates = [latest_date + timedelta(days=i+1) for i in range(prediction_days)]
    
    # Generate SIR prediction
    extended_solution = solve_ivp(
        fun=lambda t, y: enhanced_sir_model(t, y, beta_opt, gamma_opt, mu_opt, alpha_opt, population),
        t_span=[0, len(daily_data) + prediction_days],
        y0=initial_conditions,
        t_eval=np.arange(len(daily_data) + prediction_days),
        method='RK45',
        rtol=1e-6
    )
    
    S_pred, I_pred, R_pred, D_pred = extended_solution.y
    C_pred = I_pred + R_pred + D_pred  # Cumulative confirmed cases
    
    sir_r0 = beta_opt / gamma_opt
    
    return {
        'prediction_dates': prediction_dates,
        'I_pred': I_pred,
        'S_pred': S_pred,
        'R_pred': R_pred,
        'D_pred': D_pred,
        'C_pred': C_pred,
        'final_active': I_pred[-1],
        'final_recovered': R_pred[-1],
        'final_deaths': D_pred[-1],
        'final_confirmed': C_pred[-1],
        'r0': sir_r0,
        'historical_cutoff': len(daily_data),
        'latest_date': latest_date,
        'beta': beta_opt,
        'gamma': gamma_opt
    }

# Function to create SIR plot
def create_sir_plot(daily_data, sir_results, country):
    fig_sir, ax_sir = create_styled_figure(figsize=(10, 8))
    
    # Historical data + predictions for plotting
    all_dates = daily_data['Date'].tolist() + sir_results['prediction_dates']
    historical_cutoff = sir_results['historical_cutoff']
    
    # Historical data
    ax_sir.plot(all_dates[:historical_cutoff], daily_data['Active_Smooth'], 
               label='Historical Active', linewidth=2, color='#FF9900')
    ax_sir.plot(all_dates[:historical_cutoff], daily_data['Deaths_Smooth'], 
               label='Historical Deaths', linewidth=2, color='#DC3912')
    ax_sir.plot(all_dates[:historical_cutoff], daily_data['Recovered_Smooth'], 
               label='Historical Recovered', linewidth=2, color='#109618')
    
    # SIR Predictions (dotted lines)
    ax_sir.plot(all_dates[historical_cutoff-1:], sir_results['I_pred'][historical_cutoff-1:], 
               ':', label='SIR Active Prediction', linewidth=3, color='#FF9900')
    ax_sir.plot(all_dates[historical_cutoff-1:], sir_results['D_pred'][historical_cutoff-1:], 
               ':', label='SIR Deaths Prediction', linewidth=3, color='#DC3912')
    ax_sir.plot(all_dates[historical_cutoff-1:], sir_results['R_pred'][historical_cutoff-1:], 
               ':', label='SIR Recovered Prediction', linewidth=3, color='#109618')
    
    # Mark prediction start
    ax_sir.axvline(x=sir_results['latest_date'], color='black', linestyle='--', alpha=0.7, label='Prediction Start')
    
    # Format axes
    ax_sir.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_sir.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_sir.yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    ax_sir.set_title(f'SIR Model Predictions for {country}', fontweight='bold')
    ax_sir.set_xlabel('Date', fontweight='bold')
    ax_sir.set_ylabel('Number of Cases', fontweight='bold')
    ax_sir.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_sir.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_sir

# Function to create DL plot
def create_dl_plot(daily_data, dl_results, country):
    fig_dl, ax_dl = create_styled_figure(figsize=(10, 8))
    
    # Historical data
    ax_dl.plot(daily_data['Date'], daily_data['Active_Smooth'], 
              label='Historical Active', linewidth=2, color='#FF9900')
    ax_dl.plot(daily_data['Date'], daily_data['Deaths_Smooth'], 
              label='Historical Deaths', linewidth=2, color='#DC3912')
    ax_dl.plot(daily_data['Date'], daily_data['Recovered_Smooth'], 
              label='Historical Recovered', linewidth=2, color='#109618')
    
    # DL Predictions (solid lines)
    ax_dl.plot(dl_results['prediction_dates'], dl_results['predicted_active'], 
              '-', label='DL Active Prediction', linewidth=3, color='#FF9900')
    ax_dl.plot(dl_results['prediction_dates'], dl_results['predicted_deaths'], 
              '-', label='DL Deaths Prediction', linewidth=3, color='#DC3912')
    ax_dl.plot(dl_results['prediction_dates'], dl_results['predicted_recovered'], 
              '-', label='DL Recovered Prediction', linewidth=3, color='#109618')
    
    # Mark prediction start
    latest_date = daily_data['Date'].iloc[-1]
    ax_dl.axvline(x=latest_date, color='black', linestyle='--', alpha=0.7, label='Prediction Start')
    
    # Format axes
    ax_dl.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_dl.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_dl.yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    ax_dl.set_title(f'Deep Learning Predictions for {country}', fontweight='bold')
    ax_dl.set_xlabel('Date', fontweight='bold')
    ax_dl.set_ylabel('Number of Cases', fontweight='bold')
    ax_dl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_dl.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_dl

# Create two columns for models
col1, col2 = st.columns(2)

# SIR Model Column
with col1:
    st.subheader("SIR Model")
    
    # SIR model button
    if st.button("Run SIR Model", key="sir_button"):
        with st.spinner("Running SIR model..."):
            # Run SIR model
            sir_results = run_sir_model(daily_data, population, prediction_date)
            
            # Store in session state
            st.session_state['sir_results'] = sir_results
            
            # Create and store plot data
            sir_plot = create_sir_plot(daily_data, sir_results, country)
            st.session_state['sir_plot_data'] = sir_plot
            
            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Predicted Active", f"{int(sir_results['final_active']):,}")
            with metric_col2:
                st.metric("Predicted Deaths", f"{int(sir_results['final_deaths']):,}")
            with metric_col3:
                st.metric("Predicted Recovered", f"{int(sir_results['final_recovered']):,}")
            
            st.metric("Estimated R₀", f"{sir_results['r0']:.2f}")
    
    # Always display SIR plot if available
    if st.session_state['sir_plot_data'] is not None:
        st.pyplot(st.session_state['sir_plot_data'])

# Deep Learning Model Column
with col2:
    st.subheader("Deep Learning Model")
    
    # Deep Learning model button
    if st.button("Run Deep Learning Model", key="dl_button"):
        with st.spinner("Training deep learning model..."):
            # Prepare features for deep learning
            features = ['Confirmed', 'Deaths', 'Recovered', 'Active']
            
            data = daily_data[features].values
            
            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Define sequence length
            seq_length = 14  # 2 weeks for prediction
            
            # Create sequences for LSTM training for all features
            X, y = [], []
            for i in range(len(scaled_data) - seq_length):
                X.append(scaled_data[i:i+seq_length])
                y.append(scaled_data[i+seq_length])  # Predict all features
            
            X, y = np.array(X), np.array(y)
            
            # Train-test split
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build LSTM model for multi-output
            dl_model = Sequential()
            dl_model.add(LSTM(64, return_sequences=True, input_shape=(seq_length, len(features))))
            dl_model.add(Dropout(0.2))
            dl_model.add(LSTM(32, return_sequences=False))
            dl_model.add(Dropout(0.2))
            dl_model.add(Dense(len(features)))  # Output all features
            
            dl_model.compile(optimizer='adam', loss='mse')
            
            # Train the model
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)
            
            dl_model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=16,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Make predictions for future dates
            latest_date = daily_data['Date'].iloc[-1]
            prediction_days = (prediction_date - latest_date.to_pydatetime().date()).days
            
            if prediction_days <= 0:
                prediction_days = 30
            
            # Prepare the initial sequence
            last_seq = scaled_data[-seq_length:]
            
            # Make iterative predictions
            curr_seq = last_seq.copy()
            predicted_values = []
            
            for _ in range(prediction_days):
                # Reshape for prediction
                curr_seq_reshaped = curr_seq.reshape(1, seq_length, len(features))
                # Get prediction for next day
                next_pred = dl_model.predict(curr_seq_reshaped, verbose=0)[0]
                predicted_values.append(next_pred)
                
                # Update sequence for next iteration
                curr_seq = np.vstack([curr_seq[1:], next_pred])
            
            # Prepare dates for plotting
            prediction_dates = [latest_date + timedelta(days=i+1) for i in range(prediction_days)]
            
            # Inverse transform predictions to get actual values
            predicted_values = np.array(predicted_values)
            predicted_actual = scaler.inverse_transform(predicted_values)
            
            # Extract individual predictions
            predicted_confirmed = predicted_actual[:, 0]
            predicted_deaths = predicted_actual[:, 1]
            predicted_recovered = predicted_actual[:, 2]
            predicted_active = predicted_actual[:, 3]
            
            # Store predictions
            dl_results = {
                'prediction_dates': prediction_dates,
                'predicted_confirmed': predicted_confirmed,
                'predicted_deaths': predicted_deaths,
                'predicted_recovered': predicted_recovered,
                'predicted_active': predicted_active,
                'final_active': predicted_active[-1],
                'final_deaths': predicted_deaths[-1],
                'final_recovered': predicted_recovered[-1],
                'final_confirmed': predicted_confirmed[-1]
            }
            
            st.session_state['dl_results'] = dl_results
            
            # Create and store plot data
            dl_plot = create_dl_plot(daily_data, dl_results, country)
            st.session_state['dl_plot_data'] = dl_plot
            
            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Predicted Active", f"{int(predicted_active[-1]):,}")
            with metric_col2:
                st.metric("Predicted Deaths", f"{int(predicted_deaths[-1]):,}")
            with metric_col3:
                st.metric("Predicted Recovered", f"{int(predicted_recovered[-1]):,}")
    
    # Always display DL plot if available
    if st.session_state['dl_plot_data'] is not None:
        st.pyplot(st.session_state['dl_plot_data'])

# Comparison Plot
if 'sir_results' in st.session_state and 'dl_results' in st.session_state:
    st.subheader("Model Comparison")
    
    sir_results = st.session_state['sir_results']
    dl_results = st.session_state['dl_results']
    
    # Create comparison plot
    fig_comp, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Active Cases Comparison
    axes[0].plot(daily_data['Date'], daily_data['Active_Smooth'], 
                label='Historical Active', linewidth=2, color='blue')
    axes[0].plot(sir_results['prediction_dates'], sir_results['I_pred'][-len(sir_results['prediction_dates']):], 
                ':', label='SIR Active Prediction', linewidth=3, color='red')
    axes[0].plot(dl_results['prediction_dates'], dl_results['predicted_active'], 
                '-', label='DL Active Prediction', linewidth=3, color='green')
    axes[0].axvline(x=sir_results['latest_date'], color='black', linestyle='--', alpha=0.7)
    axes[0].set_title('Active Cases Comparison', fontweight='bold')
    axes[0].set_ylabel('Active Cases')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    # Deaths Comparison
    axes[1].plot(daily_data['Date'], daily_data['Deaths_Smooth'], 
                label='Historical Deaths', linewidth=2, color='blue')
    axes[1].plot(sir_results['prediction_dates'], sir_results['D_pred'][-len(sir_results['prediction_dates']):], 
                ':', label='SIR Deaths Prediction', linewidth=3, color='red')
    axes[1].plot(dl_results['prediction_dates'], dl_results['predicted_deaths'], 
                '-', label='DL Deaths Prediction', linewidth=3, color='green')
    axes[1].axvline(x=sir_results['latest_date'], color='black', linestyle='--', alpha=0.7)
    axes[1].set_title('Deaths Comparison', fontweight='bold')
    axes[1].set_ylabel('Deaths')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    # Recovered Comparison
    axes[2].plot(daily_data['Date'], daily_data['Recovered_Smooth'], 
                label='Historical Recovered', linewidth=2, color='blue')
    axes[2].plot(sir_results['prediction_dates'], sir_results['R_pred'][-len(sir_results['prediction_dates']):], 
                ':', label='SIR Recovered Prediction', linewidth=3, color='red')
    axes[2].plot(dl_results['prediction_dates'], dl_results['predicted_recovered'], 
                '-', label='DL Recovered Prediction', linewidth=3, color='green')
    axes[2].axvline(x=sir_results['latest_date'], color='black', linestyle='--', alpha=0.7)
    axes[2].set_title('Recovered Cases Comparison', fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Recovered')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    # Format date axes for all subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig_comp)
    
    # Model comparison metrics
    st.subheader("Model Performance Comparison")
    
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        st.write("**Active Cases**")
        st.write(f"SIR Model: {int(sir_results['final_active']):,}")
        st.write(f"DL Model: {int(dl_results['final_active']):,}")
        diff_active = abs(sir_results['final_active'] - dl_results['final_active'])
        st.write(f"Difference: {int(diff_active):,}")
    
    with comp_col2:
        st.write("**Deaths**")
        st.write(f"SIR Model: {int(sir_results['final_deaths']):,}")
        st.write(f"DL Model: {int(dl_results['final_deaths']):,}")
        diff_deaths = abs(sir_results['final_deaths'] - dl_results['final_deaths'])
        st.write(f"Difference: {int(diff_deaths):,}")
    
    with comp_col3:
        st.write("**Recovered**")
        st.write(f"SIR Model: {int(sir_results['final_recovered']):,}")
        st.write(f"DL Model: {int(dl_results['final_recovered']):,}")
        diff_recovered = abs(sir_results['final_recovered'] - dl_results['final_recovered'])
        st.write(f"Difference: {int(diff_recovered):,}")

# Summary Report Section
st.subheader("Prediction Summary Report")

if 'sir_results' in st.session_state and 'dl_results' in st.session_state:
    sir_results = st.session_state['sir_results']
    dl_results = st.session_state['dl_results']
    
    # Average of both models
    final_active = (sir_results['final_active'] + dl_results['final_active']) / 2
    final_deaths = (sir_results['final_deaths'] + dl_results['final_deaths']) / 2
    final_recovered = (sir_results['final_recovered'] + dl_results['final_recovered']) / 2
    
    report_col1, report_col2, report_col3 = st.columns(3)
    
    with report_col1:
        st.metric("Current Active Cases", f"{int(daily_data['Active'].iloc[-1]):,}")
        st.metric("Avg Predicted Active", f"{int(final_active):,}")
        
    with report_col2:
        st.metric("Current Deaths", f"{int(daily_data['Deaths'].iloc[-1]):,}")
        st.metric("Avg Predicted Deaths", f"{int(final_deaths):,}")
        
    with report_col3:
        st.metric("Current Recovered", f"{int(daily_data['Recovered'].iloc[-1]):,}")
        st.metric("Avg Predicted Recovered", f"{int(final_recovered):,}")
    
    # Growth calculations
    current_active = daily_data['Active'].iloc[-1]
    current_deaths = daily_data['Deaths'].iloc[-1]
    current_recovered = daily_data['Recovered'].iloc[-1]
    
    active_growth = ((final_active / max(current_active, 1)) - 1) * 100
    deaths_growth = ((final_deaths / max(current_deaths, 1)) - 1) * 100
    recovered_growth = ((final_recovered / max(current_recovered, 1)) - 1) * 100
    
    growth_col1, growth_col2, growth_col3 = st.columns(3)
    
    with growth_col1:
        st.metric("Active Growth Rate", f"{active_growth:.2f}%")
    with growth_col2:
        st.metric("Deaths Growth Rate", f"{deaths_growth:.2f}%")
    with growth_col3:
        st.metric("Recovery Growth Rate", f"{recovered_growth:.2f}%")
    
    # Summary text
    summary = f"""
    ## COVID-19 Forecast Summary for {country}
    
    ### Key Findings
    - Current Active Cases: {int(current_active):,}
    - Current Deaths: {int(current_deaths):,}
    - Current Recovered: {int(current_recovered):,}
    
    ### Projections by {prediction_date.strftime('%B %d, %Y')}
    - Predicted Active Cases: {int(final_active):,} (Growth: {active_growth:.2f}%)
    - Predicted Deaths: {int(final_deaths):,} (Growth: {deaths_growth:.2f}%)
    - Predicted Recovered: {int(final_recovered):,} (Growth: {recovered_growth:.2f}%)
    
    ### Model Predictions Comparison
    
    **SIR Model:**
    - Active: {int(sir_results['final_active']):,}
    - Deaths: {int(sir_results['final_deaths']):,}
    - Recovered: {int(sir_results['final_recovered']):,}
    - R₀: {sir_results['r0']:.2f}
    
    **Deep Learning Model:**
    - Active: {int(dl_results['final_active']):,}
    - Deaths: {int(dl_results['final_deaths']):,}
    - Recovered: {int(dl_results['final_recovered']):,}
    
    ### Model Agreement
    - Active Cases Difference: {abs(sir_results['final_active'] - dl_results['final_active']):,.0f}
    - Deaths Difference: {abs(sir_results['final_deaths'] - dl_results['final_deaths']):,.0f}
    - Recovered Difference: {abs(sir_results['final_recovered'] - dl_results['final_recovered']):,.0f}
    
    ### Interpretation
    - Active Cases: {"Projected to increase significantly" if active_growth > 10 else "Projected to increase moderately" if active_growth > 1 else "Projected to decrease" if active_growth < -1 else "Projected to remain stable"}
    - Deaths: {"Projected to increase significantly" if deaths_growth > 10 else "Projected to increase moderately" if deaths_growth > 1 else "Projected to decrease" if deaths_growth < -1 else "Projected to remain stable"}
    - Recovery: {"Projected to increase significantly" if recovered_growth > 10 else "Projected to increase moderately" if recovered_growth > 1 else "Projected to decrease" if recovered_growth < -1 else "Projected to remain stable"}
    
    ### Recommendations
    - {"Immediate public health measures recommended" if active_growth > 15 else "Enhanced monitoring and preparedness advised" if active_growth > 5 else "Continue current interventions" if active_growth > 0 else "Current interventions appear effective"}
    - {"Healthcare capacity planning required" if deaths_growth > 10 else "Monitor healthcare resources" if deaths_growth > 0 else "Healthcare burden may decrease"}
    - {"Recovery programs scaling up" if recovered_growth > 10 else "Maintain recovery support systems"}
    
    """
    
    st.markdown(summary)
    
    # Allow downloading the report
    st.download_button(
        label="Download Report as Text",
        data=summary,
        file_name=f"covid_report_{country}_{prediction_date}.txt",
        mime="text/plain",
    )
    
elif 'sir_results' in st.session_state:
    st.info("Run the Deep Learning Model to complete the summary report.")
    
elif 'dl_results' in st.session_state:
    st.info("Run the SIR Model to complete the summary report.")
    
else:
    st.info("Run both prediction models to generate the summary report.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center">
        <p>COVID-19 Analysis and Prediction Dashboard | Enhanced with Multi-Model Comparison</p>
        <p>Uses SIR Epidemiological Modeling (dotted lines) and Deep Learning LSTM (solid lines)</p>
    </div>
    """, unsafe_allow_html=True)