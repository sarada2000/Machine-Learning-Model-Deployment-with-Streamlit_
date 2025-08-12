import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide", page_icon="🚢")

# Load model once using joblib (correct for .joblib files)
@st.cache_resource
def load_model():
    return joblib.load(r'C:\Users\user\Desktop\intelligent system\model.pkl')

model = load_model()

# Load data once
@st.cache_data
def load_data():
    df = pd.read_csv(r'data\train.csv')
    return df

df = load_data()

# Sidebar with navigation and info
st.sidebar.title("🛳 Titanic Survival App")
st.sidebar.markdown("""
This app predicts whether a passenger would have survived the Titanic disaster.

Navigate:
- **Home:** App overview
- **Data Exploration:** View dataset & visuals
- **Predictions:** Input passenger info & predict survival
""")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Predictions"])

# --- HOME PAGE ---
if page == "Home":
    st.title("🚢 Titanic Survival Prediction App")
    st.markdown("""
    Welcome to the Titanic survival prediction app!  
    Explore the dataset, visualize key patterns, and predict survival outcomes based on passenger details.
    """)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg", use_column_width=True, caption="RMS Titanic - The unsinkable ship")

    st.markdown("---")
    st.header("Quick Stats Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        survived = df['Survived'].sum()
        st.metric("Total Survivors", f"{survived} passengers", delta=f"{survived / len(df):.1%} survival rate")
    with col2:
        total_passengers = len(df)
        st.metric("Total Passengers", total_passengers)
    with col3:
        died = total_passengers - survived
        st.metric("Total Deceased", died, delta=f"{died / total_passengers:.1%} death rate")

    st.markdown("---")
    st.markdown("**Explore this app using the sidebar navigation!**")

# --- DATA EXPLORATION PAGE ---
elif page == "Data Exploration":
    st.header("📊 Data Exploration & Visualization")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Summary")
    st.write(df.describe())

    st.markdown("---")
    st.subheader("Visualizations")
    
    # Survival Rate Pie Chart
    st.write("**Survival Rate Distribution**")
    fig1, ax1 = plt.subplots()
    df['Survived'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['Died', 'Survived'], colors=['#e63946', '#2a9d8f'], startangle=90, explode=[0.05,0.05], shadow=True, ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title("Survival Rate")
    st.pyplot(fig1)

    st.write("**Survival by Passenger Class**")
    fig2, ax2 = plt.subplots()
    pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', stacked=True, color=['#e63946', '#2a9d8f'], ax=ax2)
    ax2.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
    ax2.set_xlabel('Passenger Class')
    ax2.set_ylabel('Number of Passengers')
    ax2.legend(['Did Not Survive', 'Survived'])
    ax2.set_title("Survival by Class")
    st.pyplot(fig2)

    st.write("**Age Distribution of Passengers**")
    fig3, ax3 = plt.subplots()
    df['Age'].plot(kind='hist', bins=30, color='#264653', alpha=0.7, ax=ax3)
    ax3.set_xlabel('Age')
    ax3.set_title("Age Distribution")
    st.pyplot(fig3)

# --- PREDICTIONS PAGE ---
elif page == "Predictions":
    st.header("🔮 Predict Titanic Survival")
    
    st.markdown("Fill in the passenger details below and click **Predict Survival**.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = 1st class, 3 = 3rd class")
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 0, 100, 30)
        with col2:
            sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
            parch = st.slider("Parents/Children Aboard", 0, 6, 0)
            fare = st.number_input("Fare Price (in pounds)", min_value=0.0, max_value=600.0, value=50.0, step=0.1)

        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")

        submitted = st.form_submit_button("Predict Survival")

    if submitted:
        # Validate age and fare
        if age == 0:
            st.warning("Age is set to 0. This might affect prediction accuracy.")
        if fare == 0:
            st.warning("Fare is 0. This might affect prediction accuracy.")

        # Prepare input for the model
        # Note: Model expects 9 features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q, Embarked_S
        input_data = [[
            pclass,
            1 if sex == "female" else 0,
            age,
            sibsp,
            parch,
            fare,
            1 if embarked == "C" else 0,
            1 if embarked == "Q" else 0,
            1 if embarked == "S" else 0,
        ]]

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"🎉 The passenger would have SURVIVED with {proba:.0%} confidence!")
            st.balloons()
        else:
            st.error(f"☠️ The passenger would NOT have survived with {(1 - proba):.0%} confidence.")

st.markdown("---")
st.caption("Created by Sarada ❤️")










