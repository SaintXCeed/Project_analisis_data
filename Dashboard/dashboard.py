import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sns.set(style='dark')

st.title('Bike Sharing Analytics Dashboard :sparkles:')

# Create helper functions for dataframe
def create_rentals_per_year_df(df):
    rentals_per_year = df.groupby('yr')['cnt'].sum().reset_index()
    return rentals_per_year

def create_users_notholiday_and_holiday_df(df):
    users_notholiday_and_holiday = all_df.holiday.value_counts()
    return users_notholiday_and_holiday

def create_rentals_per_season_df(df):
    rentals_per_season = all_df.groupby('season')['cnt'].sum()
    return rentals_per_season

def create_users_type_df(df):
    total_casual_users = all_df['casual'].sum()
    total_registered_users = all_df['registered'].sum()

    user_type = {
        'Type of Users': ['Casual Users', 'Registered Users'],
        'Users Total': [total_casual_users, total_registered_users]
    }
    return user_type

# Load data
all_df = pd.read_csv("https://github.com/SaintXCeed/Project_analisis_data/blob/main/Dashboard/all_data.csv")
all_df["dteday"] = pd.to_datetime(all_df["dteday"])
min_date = all_df["dteday"].min()
max_date = all_df["dteday"].max()

# Sidebar for date filtering
with st.sidebar:

    st.header("Filter by Date Range")
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu', 
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
    
    # Filter data based on selected date range
    mask = (all_df['dteday'] >= pd.to_datetime(start_date)) & (all_df['dteday'] <= pd.to_datetime(end_date))
    filtered_df = all_df[mask]

# Create dataframes
rentals_per_year = create_rentals_per_year_df(filtered_df)
users_notholiday_and_holiday = create_users_notholiday_and_holiday_df(filtered_df)
rentals_per_season = create_rentals_per_season_df(filtered_df)
user_type = create_users_type_df(filtered_df)

# Tab structure
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Rentals per Year", 
                                        "Users on Holiday and Not Holiday", 
                                        "Rentals per Season", 
                                        "Casual vs Registered Users", 
                                        "Actual vs Predicted Rentals"])

# Rentals by Year
with tab1:
    st.subheader('Number of Rentals per Year')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="yr", y="cnt", data=rentals_per_year, ax=ax)
    plt.title("Number of Rentals per Year", loc="center", fontsize=15)
    plt.xlabel("Year")
    plt.ylabel("Number of Rentals")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.tick_params(axis='x', labelsize=12)
    st.pyplot(fig)

# Number of Users on Holiday and Not Holiday
with tab2:
    st.subheader('Number of Users on Holiday and Not Holiday')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(y=users_notholiday_and_holiday.values, x=users_notholiday_and_holiday.index, ax=ax)
    plt.title("Number of Users on Holiday and Not Holiday", loc="center", fontsize=15)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.tick_params(axis='x', labelsize=12)
    st.pyplot(fig)

# Number of Rentals per Season
with tab3:
    st.subheader('Number of Rentals per Season')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(y=rentals_per_season.values, x=rentals_per_season.index, ax=ax)
    plt.title("Number of Rentals per Season", loc="center", fontsize=15)
    plt.ylabel("Number of Rentals")
    plt.xlabel("Season")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.tick_params(axis='x', labelsize=12)
    st.pyplot(fig)

# User Type: Casual vs Registered
with tab4:
    st.subheader('Number of Casual Users and Registered Users')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Type of Users', y='Users Total', data=pd.DataFrame(user_type), ax=ax)
    plt.title('Number of Casual Users and Registered Users', loc="center", fontsize=15)
    plt.ylabel('Users Total')
    plt.xlabel('Type of Users')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    st.pyplot(fig)

# Actual vs Predicted Rentals
with tab5:
    st.subheader('Number of actual and predicted rentals')
    # Perform one-hot encoding on categorical columns
    all_df_encoded = pd.get_dummies(filtered_df, columns=['season', 'holiday', 'weathersit'], drop_first=True)
    
    # Redefine the features and target
    features = ['temp', 'hum', 'windspeed'] + [col for col in all_df_encoded.columns if 'season_' in col or 'holiday_' in col or 'weathersit_' in col]
    X = all_df_encoded[features]
    y = all_df_encoded['cnt']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the regression model
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # Make predictions and calculate error metrics
    y_pred = reg_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Plot actual vs predicted values
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    plt.title('Actual vs Predicted Rentals')
    plt.xlabel('Actual Rentals')
    plt.ylabel('Predicted Rentals')
    st.pyplot(fig)
