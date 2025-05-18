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
        population_df = pd.read_csv("/Users/islem/Desktop/covid 19/world_population.csv")
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

# Function to load COVID data
@st.cache_data
def load_covid_data():
    try:
        # Use the path provided in your code
        df = pd.read_csv("/Users/islem/Desktop/covid 19/covid_19_clean_complete.csv")
        
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

# Load COVID data
with st.spinner("Loading data..."):
    df = load_covid_data()
    population_dict = load_population_data()

# Sidebar for user inputs
st.sidebar.header("Settings")

# Get unique countries for dropdown
unique_countries = sorted(df['Country/Region'].unique())

# Country selection
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

# Date selection with fixed calendar
latest_date = df['Date'].max()
max_date = latest_date + timedelta(days=365)
default_date = latest_date + timedelta(days=30)  # Set default to a month ahead

# Use date_input with explicit min and max values in ISO format
prediction_date = st.sidebar.date_input(
    "Predict for date:", 
    value=default_date,
    min_value=latest_date.date() + timedelta(days=1),
    max_value=max_date.date()
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

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Data Overview", "Model Analysis", "Predictions"])

# Tab 1: Data Overview
with tab1:
    st.header(f"COVID-19 Data for {country}")
    
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
    
    # Calculate daily new cases (diff)
    daily_data['New_Confirmed'] = daily_data['Confirmed'].diff().fillna(0)
    daily_data['New_Deaths'] = daily_data['Deaths'].diff().fillna(0)
    daily_data['New_Recovered'] = daily_data['Recovered'].diff().fillna(0)
    
    # Apply rolling average to new cases
    daily_data['New_Confirmed_Smooth'] = daily_data['New_Confirmed'].rolling(window=window_size, center=False).mean().fillna(daily_data['New_Confirmed'])
    daily_data['New_Deaths_Smooth'] = daily_data['New_Deaths'].rolling(window=window_size, center=False).mean().fillna(daily_data['New_Deaths'])
    daily_data['New_Recovered_Smooth'] = daily_data['New_Recovered'].rolling(window=window_size, center=False).mean().fillna(daily_data['New_Recovered'])
    
    # Ensure no negative values in new cases (data errors)
    for col in ['New_Confirmed', 'New_Deaths', 'New_Recovered', 'New_Confirmed_Smooth', 'New_Deaths_Smooth', 'New_Recovered_Smooth']:
        daily_data[col] = daily_data[col].clip(lower=0)
    
    # Calculate additional metrics
    daily_data['Mortality_Rate'] = (daily_data['Deaths'] / daily_data['Confirmed'] * 100).fillna(0)
    daily_data['Recovery_Rate'] = (daily_data['Recovered'] / daily_data['Confirmed'] * 100).fillna(0)
    
    # Display data stats with improved formatting
    st.subheader("Current Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    latest_stats = daily_data.iloc[-1]
    
    with stats_col1:
        st.metric("Total Confirmed", f"{int(latest_stats['Confirmed']):,}", 
                 f"+{int(latest_stats['New_Confirmed']):,}")
    with stats_col2:
        st.metric("Total Deaths", f"{int(latest_stats['Deaths']):,}", 
                 f"+{int(latest_stats['New_Deaths']):,}")
    with stats_col3:
        st.metric("Total Recovered", f"{int(latest_stats['Recovered']):,}", 
                 f"+{int(latest_stats['New_Recovered']):,}")
    with stats_col4:
        st.metric("Active Cases", f"{int(latest_stats['Active']):,}")
    
    # Additional metrics
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric("Mortality Rate", f"{latest_stats['Mortality_Rate']:.2f}%")
    with metrics_col2:
        st.metric("Recovery Rate", f"{latest_stats['Recovery_Rate']:.2f}%")
    
    # Display improved charts
    st.subheader("Historical Data")
    
    # Cumulative cases chart with improved styling
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

    # Daily new cases chart with improved styling
    fig2, ax2 = create_styled_figure(figsize=(12, 5))
    
    # Create area chart for better visualization
    ax2.fill_between(daily_data['Date'], daily_data['New_Confirmed_Smooth'], 
                    alpha=0.4, color='#3366CC', label='New Confirmed (7-day avg)')
    ax2.plot(daily_data['Date'], daily_data['New_Confirmed_Smooth'], color='#3366CC', linewidth=1)
    
    ax2.fill_between(daily_data['Date'], daily_data['New_Deaths_Smooth'], 
                    alpha=0.4, color='#DC3912', label='New Deaths (7-day avg)')
    ax2.plot(daily_data['Date'], daily_data['New_Deaths_Smooth'], color='#DC3912', linewidth=1)
    
    ax2.fill_between(daily_data['Date'], daily_data['New_Recovered_Smooth'], 
                    alpha=0.4, color='#109618', label='New Recovered (7-day avg)')
    ax2.plot(daily_data['Date'], daily_data['New_Recovered_Smooth'], color='#109618', linewidth=1)
    
    # Format date axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Format y-axis for large numbers
    ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks))
    
    ax2.set_title(f'COVID-19 Daily New Cases for {country}', fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Number of Cases', fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax2.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Add visualization of rates
    fig4, ax4 = create_styled_figure(figsize=(12, 5))
    ax4.plot(daily_data['Date'], daily_data['Mortality_Rate'], label='Mortality Rate (%)', linewidth=2, color='#DC3912')
    ax4.plot(daily_data['Date'], daily_data['Recovery_Rate'], label='Recovery Rate (%)', linewidth=2, color='#109618')
    
    # Format date axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    ax4.set_title(f'COVID-19 Mortality and Recovery Rates for {country}', fontweight='bold')
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.set_ylabel('Rate (%)', fontweight='bold')
    ax4.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig4)
    
    # Show raw data with improved interface
    with st.expander("Show Raw Data"):
        st.dataframe(daily_data.style.highlight_max(subset=['New_Confirmed', 'New_Deaths'], color='red').highlight_min(subset=['New_Confirmed', 'New_Deaths'], color='green'))

# Tab 2: Model Analysis
with tab2:
    st.header("Enhanced SIR & Deep Learning Model Analysis")
    
    # Define enhanced SIR model differential equations with time-varying parameters
    def enhanced_sir_model(t, y, beta_max, beta_min, gamma_max, gamma_min, N, half_life):
        """Enhanced SIR model with time-varying parameters."""
        S, I, R = y
        
        # Time-dependent beta (infection rate decreases over time due to interventions)
        beta = beta_min + (beta_max - beta_min) * np.exp(-t / half_life)
        
        # Time-dependent gamma (recovery rate increases over time due to improved treatments)
        gamma = gamma_min + (gamma_max - gamma_min) * (1 - np.exp(-t / half_life))
        
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
    
    # Only run analysis if button is clicked
    if st.button("Run Advanced Analysis", type="primary"):
        with st.spinner("Running comprehensive analysis and training models..."):
            # Set up SIR model initial conditions
            if use_custom_sir:
                S0 = susceptible
                I0 = infected
                R0 = recovered
                st.write(f"Using custom SIR values - S: {S0:,}, I: {I0:,}, R: {R0:,}")
            else:
                I0 = daily_data['Active'].iloc[-1]
                R0 = daily_data['Recovered'].iloc[-1] + daily_data['Deaths'].iloc[-1]
                S0 = population - I0 - R0
                st.write(f"Using latest data SIR values - S: {S0:,}, I: {I0:,}, R: {R0:,}")
            
            initial_conditions = [S0, I0, R0]
            
            # SIR Model Implementation
            st.subheader("Enhanced SIR Model Fitting")
            
            # Time points
            t = np.arange(len(daily_data))
            
            # Actual data for fitting
            actual_infected = daily_data['Active_Smooth'].values
            actual_recovered = daily_data['Recovered_Smooth'].values + daily_data['Deaths_Smooth'].values
            
            # Function to minimize with advanced parameters
            def advanced_loss_function(params):
                beta_max, beta_min, gamma_max, gamma_min, half_life = params
                solution = solve_ivp(
                    fun=lambda t, y: enhanced_sir_model(t, y, beta_max, beta_min, gamma_max, gamma_min, population, half_life),
                    t_span=[0, max(t)],
                    y0=initial_conditions,
                    t_eval=t,
                    method='RK45'
                )
                
                S, I, R = solution.y
                
                # Weighted MSE loss with emphasis on recent data
                time_weights = np.exp(np.linspace(0, 1, len(t))) / np.exp(1)
                loss_I = np.mean(time_weights * ((I - actual_infected) / actual_infected.max())**2)
                loss_R = np.mean(time_weights * ((R - actual_recovered) / actual_recovered.max())**2)
                
                return loss_I + loss_R
            
            # Use differential evolution for better global optimization
            bounds = [(0.1, 2.0), (0.01, 0.5), (0.01, 0.5), (0.001, 0.1), (1, 100)]  # bounds for parameters
            
            result = differential_evolution(
                advanced_loss_function, 
                bounds, 
                strategy='best1bin', 
                popsize=15, 
                tol=0.01, 
                mutation=(0.5, 1.0),
                recombination=0.7,
                maxiter=100
            )
            
            beta_max_opt, beta_min_opt, gamma_max_opt, gamma_min_opt, half_life_opt = result.x
            
            # Calculate effective R0 (reproduction number) at the beginning and end
            initial_r0 = beta_max_opt / gamma_min_opt
            final_r0 = beta_min_opt / gamma_max_opt
            
            # Display SIR model parameters with improved presentation
            st.subheader("Optimized SIR Parameters")
            
            param_col1, param_col2, param_col3 = st.columns(3)
            with param_col1:
                st.metric("Initial β (max infection rate)", f"{beta_max_opt:.4f}")
                st.metric("Final β (min infection rate)", f"{beta_min_opt:.4f}")
            with param_col2:
                st.metric("Initial γ (min recovery rate)", f"{gamma_min_opt:.4f}")
                st.metric("Final γ (max recovery rate)", f"{gamma_max_opt:.4f}")
            with param_col3:
                st.metric("Initial R₀", f"{initial_r0:.4f}")
                st.metric("Final R₀", f"{final_r0:.4f}")
            
            st.metric("Intervention Half-life (days)", f"{half_life_opt:.2f}")
            
            # Simulate with optimized parameters
            solution_opt = solve_ivp(
                fun=lambda t, y: enhanced_sir_model(t, y, beta_max_opt, beta_min_opt, gamma_max_opt, gamma_min_opt, population, half_life_opt),
                t_span=[0, max(t)],
                y0=initial_conditions,
                t_eval=t,
                method='RK45'
            )
            
            S_opt, I_opt, R_opt = solution_opt.y
            
            # Plot enhanced SIR model fit
            fig3, ax3 = create_styled_figure(figsize=(12, 6))
            
            # Plot actual data with transparency
            ax3.scatter(daily_data['Date'], actual_infected, color='blue', alpha=0.3, s=10, label='Actual Infected')
            ax3.scatter(daily_data['Date'], actual_recovered, color='red', alpha=0.3, s=10, label='Actual Recovered+Deaths')
            
            # Plot model fit with solid lines
            ax3.plot(daily_data['Date'], I_opt, 'b-', linewidth=2.5, label='SIR Model Infected')
            ax3.plot(daily_data['Date'], R_opt, 'r-', linewidth=2.5, label='SIR Model Recovered')
            
            # Add confidence intervals (assuming 10% error)
            ax3.fill_between(daily_data['Date'], I_opt*0.9, I_opt*1.1, color='blue', alpha=0.2)
            ax3.fill_between(daily_data['Date'], R_opt*0.9, R_opt*1.1, color='red', alpha=0.2)
            
            # Format date axis
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            
            # Format y-axis for large numbers
            ax3.yaxis.set_major_formatter(FuncFormatter(format_ticks))
            
            ax3.set_title(f'Enhanced SIR Model Fit for {country}', fontweight='bold')
            ax3.set_xlabel('Date', fontweight='bold')
            ax3.set_ylabel('Cases', fontweight='bold')
            ax3.legend(loc='upper left', frameon=True, framealpha=0.9)
            ax3.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Calculate r-squared to show model accuracy
            ss_total_I = np.sum((actual_infected - np.mean(actual_infected))**2)
            ss_residual_I = np.sum((actual_infected - I_opt)**2)
            r_squared_I = 1 - (ss_residual_I / ss_total_I)
            
            ss_total_R = np.sum((actual_recovered - np.mean(actual_recovered))**2)
            ss_residual_R = np.sum((actual_recovered - R_opt)**2)
            r_squared_R = 1 - (ss_residual_R / ss_total_R)
            
            st.write(f"SIR Model Accuracy - Infected: **{r_squared_I:.4f}** (R²), Recovered: **{r_squared_R:.4f}** (R²)")
            
            # Deep Learning Model with Advanced Architecture
            st.subheader("Advanced Deep Learning Model")
            
            # Prepare features for deep learning with additional derived features
            features = ['Confirmed', 'Deaths', 'Recovered', 'Active', 
                        'New_Confirmed', 'New_Deaths', 'New_Recovered',
                        'Mortality_Rate', 'Recovery_Rate']
            
            data = daily_data[features].values
            
            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Define sequence length
            seq_length = 21  # 3 weeks for prediction (increased from 14)
            
            # Create sequences for LSTM training with multi-step output
            forecast_horizon = 7  # Predict a week ahead
            X, y = [], []
            for i in range(len(scaled_data) - seq_length - forecast_horizon + 1):
                X.append(scaled_data[i:i+seq_length])
                # Target is Active cases (index 3) for multiple days ahead
                y.append(scaled_data[i+seq_length:i+seq_length+forecast_horizon, 3])
            
            X, y = np.array(X), np.array(y)
            
            # Train-test split
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build advanced LSTM model with bidirectional layers
            dl_model = Sequential()
            dl_model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)), 
                                      input_shape=(seq_length, len(features))))
            dl_model.add(Dropout(0.3))
            dl_model.add(Bidirectional(LSTM(64, return_sequences=False)))
            dl_model.add(Dropout(0.3))
            dl_model.add(Dense(32, activation='relu'))
            dl_model.add(Dropout(0.2))
            dl_model.add(Dense(forecast_horizon))  # Multi-step output
            
            # Use Adam optimizer with learning rate schedule
            optimizer = Adam(learning_rate=0.001)
            dl_model.compile(optimizer=optimizer, loss='mse')
            
            # Early stopping and learning rate reduction
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True)
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            
            # Train the model with more epochs
            with st.spinner('Training advanced deep learning model...'):
                history = dl_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
            
            st.success("Model training completed successfully!")
            
            # Plot training history with improved visual
            fig5, ax5 = create_styled_figure(figsize=(10, 4))
            ax5.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#3366CC')
            ax5.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#DC3912')
            
            # Add markers at important points
            min_val_loss_idx = np.argmin(history.history['val_loss'])
            min_val_loss = history.history['val_loss'][min_val_loss_idx]
            ax5.scatter(min_val_loss_idx, min_val_loss, color='red', s=100, zorder=5)
            ax5.annotate(f'Best model\nLoss: {min_val_loss:.4f}', 
                        (min_val_loss_idx, min_val_loss),
                        xytext=(10, -30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='black'))
            
            ax5.set_title('Deep Learning Model Training Performance', fontweight='bold')
            ax5.set_xlabel('Epoch', fontweight='bold')
            ax5.set_ylabel('Loss (MSE)', fontweight='bold')
            ax5.legend(loc='upper right', frameon=True, framealpha=0.9)
            ax5.set_yscale('log')  # Log scale for better visualization
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig5)
            
            # Evaluate model on test data
            y_pred = dl_model.predict(X_test)
            mse = np.mean((y_pred - y_test)**2)
            rmse = np.sqrt(mse)
            
            # Calculate normalized RMSE
            y_test_inv = scaler.inverse_transform(np.hstack([
                np.zeros((y_test.shape[0], 3)),  # Placeholder for other features
                y_test[:, 0].reshape(-1, 1),     # First day prediction
                np.zeros((y_test.shape[0], len(features)-4))  # Placeholder for remaining features
            ]))[:, 3]  # Extract the active cases column
            
            y_test_max = np.max(y_test_inv)
            normalized_rmse = rmse / y_test_max * 100
            
            st.metric("Deep Learning Model RMSE", f"{rmse:.2f}")
            st.metric("Normalized RMSE (%)", f"{normalized_rmse:.2f}%")
            
            # Store the model parameters for prediction
            st.session_state['sir_params'] = (beta_max_opt, beta_min_opt, gamma_max_opt, gamma_min_opt, half_life_opt)
            st.session_state['dl_model'] = dl_model
            st.session_state['scaler'] = scaler
            st.session_state['initial_conditions'] = initial_conditions
            st.session_state['daily_data'] = daily_data
            st.session_state['population'] = population
            st.session_state['seq_length'] = seq_length
            st.session_state['features'] = features
            st.session_state['forecast_horizon'] = forecast_horizon
            st.session_state['analysis_complete'] = True
            
            st.success("Advanced analysis completed! Go to the Predictions tab to see forecasts.")
    else:
        st.info("Click 'Run Advanced Analysis' to fit enhanced models to the data.")

# Tab 3: Predictions
with tab3:
    st.header(f"COVID-19 Predictions for {country}")
    
    if st.session_state.get('analysis_complete', False):
        # Define weights for hybrid model with more options
        st.subheader("Hybrid Model Settings")
        
        model_type = st.radio(
            "Model Approach",
            ["Weighted Ensemble", "Dynamic Weighting", "Boosted Ensemble"],
            horizontal=True
        )
        
        if model_type == "Weighted Ensemble":
            # Simple weighted ensemble
            sir_weight = st.slider("SIR Model Weight", 0.0, 1.0, 0.4, 0.1)
            dl_weight = 1.0 - sir_weight
            st.write(f"Deep Learning Model Weight: {dl_weight:.1f}")
            weights_method = "fixed"
        
        elif model_type == "Dynamic Weighting":
            st.write("Weights will be determined dynamically based on recent model performance")
            # Start with equal weights as default
            sir_weight = 0.5
            dl_weight = 0.5
            weights_method = "dynamic"
        
        else:  # Boosted Ensemble
            st.write("Using gradient boosting to combine model predictions")
            sir_weight = 0.5  # Initial weights before boosting
            dl_weight = 0.5
            weights_method = "boosted"
            
        # Confidence interval settings
        show_confidence = st.checkbox("Show prediction confidence intervals", value=True)
        
        if show_confidence:
            confidence_level = st.select_slider(
                "Confidence Level",
                options=[80, 85, 90, 95, 99],
                value=90
            )
        
        # Get parameters from session state
        sir_params = st.session_state['sir_params']
        dl_model = st.session_state['dl_model']
        scaler = st.session_state['scaler']
        initial_conditions = st.session_state['initial_conditions']
        daily_data = st.session_state['daily_data']
        population = st.session_state['population']
        seq_length = st.session_state['seq_length']
        features = st.session_state['features']
        forecast_horizon = st.session_state.get('forecast_horizon', 7)
        
        # Calculate days between last data point and target date
        latest_date = daily_data['Date'].iloc[-1]
        prediction_days = (prediction_date - latest_date.to_pydatetime().date()).days
        
        if prediction_days <= 0:
            st.warning(f"Target date {prediction_date} is not in the future. Using next day instead.")
            prediction_days = 1
        
        # Make prediction button
        if st.button("Generate Advanced Predictions", type="primary"):
            with st.spinner(f"Generating predictions for {prediction_days} days ahead..."):
                # Unpack SIR parameters
                beta_max_opt, beta_min_opt, gamma_max_opt, gamma_min_opt, half_life_opt = sir_params
                
                # Initialize prediction results
                prediction_dates = [latest_date + timedelta(days=i+1) for i in range(prediction_days)]
                predictions = []
                
                # For confidence intervals
                n_simulations = 50
                all_active_predictions = np.zeros((n_simulations, prediction_days))
                all_death_predictions = np.zeros((n_simulations, prediction_days))
                all_recovered_predictions = np.zeros((n_simulations, prediction_days))
                
                # Extract current state
                current_S, current_I, current_R = initial_conditions
                
                latest_confirmed = daily_data['Confirmed'].iloc[-1]
                latest_deaths = daily_data['Deaths'].iloc[-1]
                latest_recovered = daily_data['Recovered'].iloc[-1]
                
                # Get mortality and recovery rates from recent data
                recent_window = min(30, len(daily_data))  # Last 30 days or all data if less
                recent_data = daily_data.tail(recent_window)
                
                # Calculate rates with exponential weighted moving average for higher importance on recent data
                mortality_rate = recent_data['New_Deaths'].ewm(span=7).mean().iloc[-1] / recent_data['New_Confirmed'].ewm(span=7).mean().iloc[-1]
                recovery_rate = recent_data['New_Recovered'].ewm(span=7).mean().iloc[-1] / recent_data['New_Confirmed'].ewm(span=7).mean().iloc[-1]
                
                # Handle NaN values
                mortality_rate = 0.02 if np.isnan(mortality_rate) else min(mortality_rate, 0.15)  # Cap at reasonable max
                recovery_rate = 0.1 if np.isnan(recovery_rate) else min(recovery_rate, 0.3)  # Cap at reasonable max
                
                # Get deep learning input
                recent_features = daily_data[features].tail(seq_length).values
                scaled_recent = scaler.transform(recent_features)
                
                # Dynamic weighting setup - use recent performance to determine weights
                if weights_method == "dynamic":
                    # Get last 14 days predictions from both models and compare to actual
                    days_to_check = min(14, len(daily_data) - seq_length)
                    
                    # SIR model error for last days_to_check days
                    sir_errors = []
                    dl_errors = []
                    
                    for i in range(days_to_check):
                        # Get actual value for day i
                        actual_active = daily_data['Active'].iloc[-(i+1)]
                        
                        # SIR model prediction
                        temp_solution = solve_ivp(
                            fun=lambda t, y: enhanced_sir_model(t, y, beta_max_opt, beta_min_opt, 
                                                               gamma_max_opt, gamma_min_opt, 
                                                               population, half_life_opt),
                            t_span=[0, 1],
                            y0=[daily_data['Active'].iloc[-(i+2)], daily_data['Recovered'].iloc[-(i+2)]],
                            t_eval=[0, 1],
                            method='RK45'
                        )
                        sir_pred = temp_solution.y[1][1]  # Infected component
                        sir_errors.append((sir_pred - actual_active)**2)
                        
                        # DL model prediction
                        temp_data = daily_data[features].iloc[-(seq_length+i+1):-(i+1)].values
                        temp_scaled = scaler.transform(temp_data)
                        dl_input = temp_scaled.reshape(1, seq_length, len(features))
                        dl_pred_scaled = dl_model.predict(dl_input, verbose=0)[0][0]  # First day prediction
                        
                        # Handle inverse transform
                        dummy = np.zeros((1, len(features)))
                        dummy[0, 3] = dl_pred_scaled  # Active is at index 3
                        dl_pred = scaler.inverse_transform(dummy)[0, 3]
                        
                        dl_errors.append((dl_pred - actual_active)**2)
                    
                    # Calculate average errors
                    sir_mse = np.mean(sir_errors)
                    dl_mse = np.mean(dl_errors)
                    
                    # Calculate weights inversely proportional to errors
                    total_error = sir_mse + dl_mse
                    if total_error > 0:
                        sir_weight = dl_mse / total_error  # Higher weight for model with lower error
                        dl_weight = sir_mse / total_error
                    else:
                        sir_weight = 0.5
                        dl_weight = 0.5
                    
                    st.write(f"Dynamically determined weights - SIR: {sir_weight:.2f}, Deep Learning: {dl_weight:.2f}")
                
                # For boosted ensemble, we'll need to train a simple meta-model
                if weights_method == "boosted":
                    # For simplicity, use a linear regression model as the meta-learner
                    from sklearn.linear_model import Ridge
                    
                    # Get historical predictions from both models
                    train_size = min(60, len(daily_data) - seq_length)  # Use up to 60 days of data
                    
                    meta_X = []  # Will contain [sir_pred, dl_pred] pairs
                    meta_y = []  # Will contain actual values
                    
                    for i in range(train_size):
                        # Get actual value
                        actual_active = daily_data['Active'].iloc[-(i+1)]
                        meta_y.append(actual_active)
                        
                        # Get SIR prediction
                        day_index = len(daily_data) - (i+1)
                        past_solution = solve_ivp(
                            fun=lambda t, y: enhanced_sir_model(t, y, beta_max_opt, beta_min_opt, 
                                                              gamma_max_opt, gamma_min_opt, 
                                                              population, half_life_opt),
                            t_span=[0, day_index],
                            y0=initial_conditions,
                            t_eval=[day_index-1, day_index],
                            method='RK45'
                        )
                        sir_pred = past_solution.y[1][1]
                        
                        # Get DL prediction
                        if i+seq_length < len(daily_data):
                            temp_data = daily_data[features].iloc[-(seq_length+i+1):-(i+1)].values
                            temp_scaled = scaler.transform(temp_data)
                            dl_input = temp_scaled.reshape(1, seq_length, len(features))
                            dl_pred_scaled = dl_model.predict(dl_input, verbose=0)[0][0]
                            
                            # Handle inverse transform
                            dummy = np.zeros((1, len(features)))
                            dummy[0, 3] = dl_pred_scaled  # Active is at index 3
                            dl_pred = scaler.inverse_transform(dummy)[0, 3]
                        else:
                            dl_pred = daily_data['Active'].iloc[-(i+1)]  # Fallback
                        
                        meta_X.append([sir_pred, dl_pred])
                    
                    # Convert to arrays
                    meta_X = np.array(meta_X)
                    meta_y = np.array(meta_y)
                    
                    # Train meta-model with regularization
                    meta_model = Ridge(alpha=1.0)
                    meta_model.fit(meta_X, meta_y)
                    
                    st.write(f"Meta-model coefficients - SIR: {meta_model.coef_[0]:.3f}, Deep Learning: {meta_model.coef_[1]:.3f}")
                    
                    # Normalize coefficients to get weights (ensure they're positive)
                    coefs = np.abs(meta_model.coef_)
                    sir_weight = coefs[0] / np.sum(coefs)
                    dl_weight = coefs[1] / np.sum(coefs)
                    
                    st.write(f"Boosted ensemble weights - SIR: {sir_weight:.2f}, Deep Learning: {dl_weight:.2f}")
                
                # Function to get DL prediction for multiple steps
                def get_dl_predictions(input_data, steps):
                    """Get multi-step predictions from DL model"""
                    predictions = []
                    current_input = input_data.copy()
                    
                    for _ in range(0, steps, forecast_horizon):
                        # Reshape for LSTM input
                        dl_input = current_input.reshape(1, seq_length, len(features))
                        
                        # Get predictions for forecast_horizon steps
                        step_preds = dl_model.predict(dl_input, verbose=0)[0]
                        
                        # Add predictions to results
                        num_steps = min(forecast_horizon, steps - len(predictions))
                        predictions.extend(step_preds[:num_steps])
                        
                        if len(predictions) >= steps:
                            break
                        
                        # Update input for next prediction
                        # Remove oldest entries
                        current_input = current_input[num_steps:]
                        
                        # Add new predictions (with dummy data for other features)
                        for p in step_preds[:num_steps]:
                            # Create a row with zeros
                            new_row = np.zeros((1, len(features)))
                            # Set the active cases value (index 3)
                            new_row[0, 3] = p
                            # Append to input
                            current_input = np.vstack([current_input, new_row])
                    
                    return predictions[:steps]
                
                # Generate Monte Carlo simulations for uncertainty estimation
                for sim in range(n_simulations):
                    # Add noise to parameters
                    if sim > 0:  # First simulation uses exact parameters
                        noise_factor = 0.05  # 5% noise
                        noisy_sir_params = (
                            beta_max_opt * (1 + np.random.normal(0, noise_factor)),
                            beta_min_opt * (1 + np.random.normal(0, noise_factor)),
                            gamma_max_opt * (1 + np.random.normal(0, noise_factor)),
                            gamma_min_opt * (1 + np.random.normal(0, noise_factor)),
                            half_life_opt * (1 + np.random.normal(0, noise_factor/2))
                        )
                    else:
                        noisy_sir_params = sir_params
                    
                    # Initialize for this simulation
                    sim_S, sim_I, sim_R = initial_conditions
                    sim_confirmed = latest_confirmed
                    sim_deaths = latest_deaths
                    sim_recovered = latest_recovered
                    
                    # Deep learning input with slight noise for Monte Carlo
                    if sim > 0:
                        noisy_recent = scaled_recent.copy()
                        noisy_recent += np.random.normal(0, 0.02, noisy_recent.shape)  # Add 2% noise
                    else:
                        noisy_recent = scaled_recent.copy()
                    
                    # Get DL predictions for all steps at once
                    dl_preds_scaled = get_dl_predictions(noisy_recent, prediction_days)
                    
                    # Process predictions day by day
                    for i in range(prediction_days):
                        # Unpack SIR parameters
                        noisy_beta_max, noisy_beta_min, noisy_gamma_max, noisy_gamma_min, noisy_half_life = noisy_sir_params
                        
                        # SIR prediction for next day
                        next_day_solution = solve_ivp(
                            fun=lambda t, y: enhanced_sir_model(t, y, noisy_beta_max, noisy_beta_min, 
                                                              noisy_gamma_max, noisy_gamma_min, 
                                                              population, noisy_half_life),
                            t_span=[0, 1],
                            y0=[sim_S, sim_I, sim_R],
                            t_eval=[0, 1],
                            method='RK45'
                        )
                        
                        S_sir, I_sir, R_sir = next_day_solution.y
                        sir_pred_active = I_sir[1]  # Next day prediction
                        
                        # Get the corresponding DL prediction
                        dl_pred_scaled = dl_preds_scaled[i]
                        
                        # Handle inverse transform for DL prediction
                        dummy = np.zeros((1, len(features)))
                        dummy[0, 3] = dl_pred_scaled  # Active is at index 3
                        dl_pred_active = scaler.inverse_transform(dummy)[0, 3]
                        
                        # Combine predictions
                        if weights_method == "boosted":
                            # Use meta-model for prediction
                            hybrid_pred_active = meta_model.predict([[sir_pred_active, dl_pred_active]])[0]
                        else:
                            # Use weighted average
                            hybrid_pred_active = sir_weight * sir_pred_active + dl_weight * dl_pred_active
                        
                        # Ensure prediction is reasonable
                        hybrid_pred_active = max(0, min(hybrid_pred_active, population))
                        
                        # New cases estimate with slight randomness for Monte Carlo
                        if sim > 0:
                            rand_factor = np.random.normal(1, 0.1)  # 10% randomness
                        else:
                            rand_factor = 1.0
                        
                        new_cases = max(0, hybrid_pred_active - sim_I) * rand_factor
                        
                        # Update total counts with noise for Monte Carlo
                        if sim > 0:
                            mortality_noise = np.random.normal(1, 0.05)  # 5% noise
                            recovery_noise = np.random.normal(1, 0.05)  # 5% noise
                        else:
                            mortality_noise = 1.0
                            recovery_noise = 1.0
                        
                        next_confirmed = sim_confirmed + new_cases
                        next_deaths = sim_deaths + (new_cases * mortality_rate * mortality_noise)
                        next_recovered = sim_recovered + (new_cases * recovery_rate * recovery_noise)
                        
                        # Store predictions for this simulation
                        all_active_predictions[sim, i] = hybrid_pred_active
                        all_death_predictions[sim, i] = next_deaths
                        all_recovered_predictions[sim, i] = next_recovered
                        
                        # Update for next iteration
                        sim_S = S_sir[1]
                        sim_I = hybrid_pred_active
                        sim_R = R_sir[1]
                        
                        sim_confirmed = next_confirmed
                        sim_deaths = next_deaths
                        sim_recovered = next_recovered
                
                # Calculate mean predictions and confidence intervals
                mean_active = np.mean(all_active_predictions, axis=0)
                mean_deaths = np.mean(all_death_predictions, axis=0)
                mean_recovered = np.mean(all_recovered_predictions, axis=0)
                
                # Calculate confidence intervals
                conf_interval = (100 - confidence_level) / 2.0 / 100
                lower_active = np.percentile(all_active_predictions, conf_interval * 100, axis=0)
                upper_active = np.percentile(all_active_predictions, (1 - conf_interval) * 100, axis=0)
                
                lower_deaths = np.percentile(all_death_predictions, conf_interval * 100, axis=0)
                upper_deaths = np.percentile(all_death_predictions, (1 - conf_interval) * 100, axis=0)
                
                lower_recovered = np.percentile(all_recovered_predictions, conf_interval * 100, axis=0)
                upper_recovered = np.percentile(all_recovered_predictions, (1 - conf_interval) * 100, axis=0)
                
                # Create final predictions dataframe
                pred_df = pd.DataFrame({
                    'Date': prediction_dates,
                    'Confirmed': sim_confirmed,
                    'Active': mean_active,
                    'Active_Lower': lower_active,
                    'Active_Upper': upper_active,
                    'Deaths': mean_deaths,
                    'Deaths_Lower': lower_deaths,
                    'Deaths_Upper': upper_deaths,
                    'Recovered': mean_recovered,
                    'Recovered_Lower': lower_recovered,
                    'Recovered_Upper': upper_recovered
                })
                
                # Display final prediction
                final_pred = pred_df.iloc[-1]
                
                st.subheader(f"Prediction for {prediction_date}")
                
                final_col1, final_col2, final_col3, final_col4 = st.columns(4)
                with final_col1:
                    st.metric("Confirmed Cases", f"{int(final_pred['Confirmed']):,}")
                with final_col2:
                    st.metric("Active Cases", f"{int(final_pred['Active']):,}", 
                             f"±{int(final_pred['Active_Upper'] - final_pred['Active']):,}")
                with final_col3:
                    st.metric("Deaths", f"{int(final_pred['Deaths']):,}", 
                             f"±{int(final_pred['Deaths_Upper'] - final_pred['Deaths']):,}")
                with final_col4:
                    st.metric("Recovered", f"{int(final_pred['Recovered']):,}", 
                             f"±{int(final_pred['Recovered_Upper'] - final_pred['Recovered']):,}")
                
                # Create combined historical and prediction data for plotting
                historical_dates = daily_data['Date'].tolist()
                historical_active = daily_data['Active'].tolist()
                historical_deaths = daily_data['Deaths'].tolist()
                historical_recovered = daily_data['Recovered'].tolist()
                
                pred_dates = pred_df['Date'].tolist()
                
                # Plot predictions with enhanced visualizations
                st.subheader("Predictions Visualization")
                
                # Create plot with confidence intervals
                fig6, ax6 = create_styled_figure(figsize=(14, 7))
                
                # Historical data
                ax6.plot(historical_dates, historical_active, color='#3366CC', 
                        linewidth=2.5, label='Historical Active Cases')
                ax6.plot(historical_dates, historical_deaths, color='#DC3912', 
                        linewidth=2.5, label='Historical Deaths')
                ax6.plot(historical_dates, historical_recovered, color='#109618', 
                        linewidth=2.5, label='Historical Recovered')
                
                # Predictions
                ax6.plot(pred_dates, pred_df['Active'], color='#3366CC', 
                        linestyle='--', linewidth=2.5, label='Predicted Active Cases')
                ax6.plot(pred_dates, pred_df['Deaths'], color='#DC3912', 
                        linestyle='--', linewidth=2.5, label='Predicted Deaths')
                ax6.plot(pred_dates, pred_df['Recovered'], color='#109618', 
                        linestyle='--', linewidth=2.5, label='Predicted Recovered')
                
                # Add confidence intervals if enabled
                if show_confidence:
                    ax6.fill_between(pred_dates, pred_df['Active_Lower'], pred_df['Active_Upper'],
                                   color='#3366CC', alpha=0.2, label=f'{confidence_level}% CI Active')
                    ax6.fill_between(pred_dates, pred_df['Deaths_Lower'], pred_df['Deaths_Upper'],
                                   color='#DC3912', alpha=0.2, label=f'{confidence_level}% CI Deaths')
                    ax6.fill_between(pred_dates, pred_df['Recovered_Lower'], pred_df['Recovered_Upper'],
                                   color='#109618', alpha=0.2, label=f'{confidence_level}% CI Recovered')
                
                # Highlight transition point
                ax6.axvline(x=latest_date, color='black', linestyle='--', alpha=0.7, label='Forecast Start')
                
                # Format date axis
                ax6.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
                ax6.xaxis.set_major_locator(mdates.DayLocator(interval=max(prediction_days // 10, 1)))
                
                # Format y-axis for large numbers
                ax6.yaxis.set_major_formatter(FuncFormatter(format_ticks))
                
                ax6.set_title(f'COVID-19 Forecast for {country} through {prediction_date}', fontweight='bold')
                ax6.set_xlabel('Date', fontweight='bold')
                ax6.set_ylabel('Cases', fontweight='bold')
                ax6.legend(loc='upper left', frameon=True, framealpha=0.9)
                ax6.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig6)
                
                # Add secondary plot showing daily increases
                fig8, ax8 = create_styled_figure(figsize=(14, 5))
                
                # Calculate daily changes
                pred_df['New_Active'] = np.append(
                    [pred_df['Active'][0] - daily_data['Active'].iloc[-1]], 
                    np.diff(pred_df['Active'])
                )
                pred_df['New_Deaths'] = np.append(
                    [pred_df['Deaths'][0] - daily_data['Deaths'].iloc[-1]], 
                    np.diff(pred_df['Deaths'])
                )
                pred_df['New_Recovered'] = np.append(
                    [pred_df['Recovered'][0] - daily_data['Recovered'].iloc[-1]], 
                    np.diff(pred_df['Recovered'])
                )
                
                # Ensure no negative values
                for col in ['New_Active', 'New_Deaths', 'New_Recovered']:
                    pred_df[col] = pred_df[col].clip(lower=0)
                
                # Plot daily changes
                ax8.bar(pred_dates, pred_df['New_Active'], color='#3366CC', alpha=0.7, 
                      label='Daily Change in Active Cases')
                ax8.bar(pred_dates, pred_df['New_Deaths'], color='#DC3912', alpha=0.7, 
                      label='Daily New Deaths')
                ax8.bar(pred_dates, pred_df['New_Recovered'], color='#109618', alpha=0.7, 
                      label='Daily New Recovered')
                
                # Format date axis
                ax8.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
                ax8.xaxis.set_major_locator(mdates.DayLocator(interval=max(prediction_days // 10, 1)))
                
                # Format y-axis for large numbers
                ax8.yaxis.set_major_formatter(FuncFormatter(format_ticks))
                
                ax8.set_title(f'Predicted Daily Changes for {country}', fontweight='bold')
                ax8.set_xlabel('Date', fontweight='bold')
                ax8.set_ylabel('Daily Cases', fontweight='bold')
                ax8.legend(loc='upper right', frameon=True, framealpha=0.9)
                ax8.grid(True, axis='y', alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig8)
                
                # Add statistical analysis of the predictions
                st.subheader("Prediction Statistics")
                
                # Calculate key statistics
                peak_active = max(pred_df['Active'])
                peak_active_date = pred_df.loc[pred_df['Active'].idxmax(), 'Date']
                
                total_new_deaths = sum(pred_df['New_Deaths'])
                avg_daily_cases = pred_df['New_Active'].mean()
                
                # Growth rate calculations
                current_active = daily_data['Active'].iloc[-1]
                final_active = pred_df['Active'].iloc[-1]
                
                if current_active > 0:
                    growth_rate = ((final_active / current_active) ** (1/prediction_days) - 1) * 100
                else:
                    growth_rate = 0
                
                # Display statistics
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Average Daily New Cases", f"{int(avg_daily_cases):,}")
                with stats_col2:
                    st.metric("Total Projected New Deaths", f"{int(total_new_deaths):,}")
                with stats_col3:
                    if peak_active > daily_data['Active'].iloc[-1]:
                        st.metric("Peak Active Cases", f"{int(peak_active):,}", 
                                 f"on {peak_active_date.strftime('%b %d')}")
                    else:
                        st.metric("Peak Active Cases", "Already passed")
                with stats_col4:
                    st.metric("Daily Growth Rate", f"{growth_rate:.2f}%")
                
                # Show prediction data with improved styling
                with st.expander("Show Detailed Prediction Data"):
                    # Format data for display
                    display_df = pred_df[['Date', 'Active', 'Deaths', 'Recovered', 'New_Active', 'New_Deaths', 'New_Recovered']]
                    display_df = display_df.round(0).astype({col: int for col in display_df.columns if col != 'Date'})
                    
                    st.dataframe(display_df.style.highlight_max(subset=['New_Active'], color='#ffcccc')
                                           .bar(subset=['Active'], color='#3366CC')
                                           .bar(subset=['Deaths'], color='#DC3912')
                                           .bar(subset=['Recovered'], color='#109618'))
                
                # Option to download predictions
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="Download Detailed Predictions as CSV",
                    data=csv,
                    file_name=f"covid_predictions_{country}_{prediction_date}.csv",
                    mime="text/csv",
                )
                
                # Add summary report
                st.subheader("Prediction Summary Report")
                
                summary = f"""
                ## COVID-19 Forecast Summary for {country}
                
                ### Forecast Period
                - **From:** {latest_date.strftime('%B %d, %Y')}
                - **To:** {prediction_date.strftime('%B %d, %Y')} ({prediction_days} days)
                
                ### Key Findings
                - **Current Active Cases:** {int(daily_data['Active'].iloc[-1]):,}
                - **Projected Active Cases by {prediction_date}:** {int(final_pred['Active']):,} (±{int(final_pred['Active_Upper'] - final_pred['Active']):,})
                - **Projected Total Deaths by {prediction_date}:** {int(final_pred['Deaths']):,} (±{int(final_pred['Deaths_Upper'] - final_pred['Deaths']):,})
                - **Daily Growth Rate:** {growth_rate:.2f}%
                
                ### Modeling Approach
                - Hybrid model using enhanced SIR epidemiological model and Deep Learning
                - {model_type} with SIR weight: {sir_weight:.2f}, Deep Learning weight: {dl_weight:.2f}
                - Confidence intervals calculated using Monte Carlo simulation ({confidence_level}% confidence level)
                
                ### Recommendations
                - {"Maintain current interventions as the situation appears stable" if abs(growth_rate) < 1 else "Consider strengthening interventions as cases are projected to increase significantly" if growth_rate > 0 else "Current interventions are effective as cases are projected to decrease"}
                - Continue monitoring for changes in transmission patterns
                - Update predictions regularly as new data becomes available
                """
                
                st.markdown(summary)
    else:
        st.info("Please complete the analysis in the 'Model Analysis' tab first.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center">
        <p>COVID-19 Analysis and Prediction Dashboard | Created with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)