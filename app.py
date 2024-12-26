import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="PCOS Prediction And Analysis", page_icon="🌞", layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_data():
    data = pd.read_csv("C:\\Users\\hp\\Desktop\\IDS PROJECT\\CLEAN- PCOS SURVEY SPREADSHEET.csv")
    data.columns = ['Age', 'Weight (in Kg)', 'Height', 'bloodGroup', 
                    'MenstrualCycleRegularity', 'RecentWeightChange', 'Hirsutism',
                    'SkinDarkening', 'HairLoss', 'Acne', 'FastFoodConsumption',
                    'ExerciseRegularity', 'PCOS', 'MoodSwings', 'PeriodDuration']
    return data
data = load_data()

st.markdown("""
    <style>
        .main { background-color: #f0f8ff; padding: 20px; font-family: 'Arial', sans-serif; }
        .sidebar .sidebar-content { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
        h1 { color: #2c3e50; font-size: 28px; font-weight: bold; margin-bottom: 10px; }
        h2, h3 { color: #34495e; font-weight: bold; }
        p { font-size: 16px; color: #555555; }
        .footer { margin-top: 20px; text-align: center; font-size: 14px; color: #777777; }
        .card { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
        .metric { display: flex; justify-content: space-around; margin-bottom: 20px; }
        .metric > div { text-align: center; padding: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌞 PCOS Prediction And Analysis")
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! Let's get started with the PCOS Prediction and Analysis.")
    st.write("Dive into the factors influencing PCOS and interact with predictive models for personalized insights.")
    st.sidebar.header("PCOS Analysis and Prediction")
    options = [
        "Introduction",
        "Data Overview",
        "Exploratory Data Analysis (EDA)",
        "Model Training and Evaluation",
        "Prediction on New Data",
        "Conclusion and Insights"
    ]
    selected_option = st.sidebar.radio("Select a section to explore:", options)
    show_details = st.sidebar.checkbox("Show Key Objectives of This Application")

    if selected_option == "Introduction":
        st.header("🔍 Introduction")
        st.markdown(""" 
        Polycystic Ovary Syndrome (PCOS) is one of the most prevalent endocrine disorders affecting women of reproductive age, with significant implications for reproductive, metabolic, and psychological health. It can result in symptoms such as irregular menstrual cycles, excessive androgen levels, and the presence of multiple cysts in the ovaries. If left undiagnosed or unmanaged, PCOS can lead to severe complications such as infertility, diabetes, cardiovascular diseases, and mental health challenges.  
        In this project, we delve into a dataset collected from PCOS-related surveys and medical assessments. By leveraging machine learning models, we aim to predict the likelihood of PCOS based on various health parameters such as age, weight, menstrual cycle regularity, and other factors. This tool not only highlights the importance of early diagnosis but also provides insights into the key factors contributing to the condition.
        This application is designed to serve as an educational and analytical tool to raise awareness about PCOS, support healthcare professionals, and empower women to understand their health better. By making the information accessible and interactive, we hope to contribute to proactive health management and early intervention strategies.
        """)
        image_path = r"C:\Users\hp\Desktop\IDS PROJECT\35b69288667953.Y3JvcCwxNDAwLDEwOTUsMCw5Mw.png"
        st.image(image_path, caption="PCOS Awareness", width=500) 
        if show_details:
            st.markdown("""
            <div class="card">
                <h3>Key Objectives of This Application:</h3>
                <ol>
                    <li><strong>Exploratory Data Analysis (EDA):</strong> Analyze the dataset to uncover trends, correlations, and significant features linked to PCOS.</li>
                    <li><strong>Machine Learning Modeling:</strong> Train and evaluate classification models to predict PCOS with high accuracy.</li>
                    <li><strong>Feature Importance:</strong> Identify the most critical factors influencing PCOS predictions through feature selection techniques.</li>
                    <li><strong>Interactive Prediction:</strong> Allow users to input their health data and receive a personalized prediction regarding their PCOS risk.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

    if selected_option == "Data Overview":
        st.header("🗂️ Dataset Overview")

        st.subheader("Attributes in the Dataset:")
        attribute_names = data.columns.tolist()
        attribute_df = pd.DataFrame(attribute_names, columns=["Attribute Names"])
        st.table(attribute_df)

        total_rows = len(data)
        st.write(f"Total number of rows in the dataset: {total_rows}")

        st.subheader("Summary Statistics for Numerical Columns:")
        numerical_columns = ['Age', 'Weight (in Kg)', 'Height', 'PeriodDuration']
        summary_stats = data[numerical_columns].agg(['min', 'max', 'mean', 'count']).T
        summary_stats.columns = ['Min', 'Max', 'Average', 'Count']
        st.write(summary_stats)

        st.subheader("📊 Dataset Preview")
        st.dataframe(data.head(15))  

        st.subheader("Dataset Information:")
        info_df = pd.DataFrame({
            'Non-Null Count': data.notnull().sum(),
            'Data Type': data.dtypes
        })
        st.table(info_df)

    if selected_option == "Exploratory Data Analysis (EDA)":
        st.header("🔍 Exploratory Data Analysis (EDA)")

        st.subheader("1️⃣ Correlation Matrix")
        numerical_columns = ['Age', 'Weight (in Kg)', 'Height', 'PeriodDuration']
        correlation_matrix = data[numerical_columns].corr()
        st.write("Correlation Matrix Table")
        st.dataframe(correlation_matrix)

        st.write("Correlation Matrix Heatmap")
        plt.figure(figsize=(5, 3))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        st.pyplot(plt)

        st.header("2️⃣ Column-wise Count Visualization")
        selected_column = st.selectbox("Select a column to visualize counts:", data.columns)

        if selected_column:
            value_counts = data[selected_column].value_counts()
            st.subheader(f"Counts for {selected_column}:")
            st.table(value_counts)

            st.subheader(f"Bar Chart for {selected_column}:")
            fig, ax = plt.subplots(figsize=(5, 3))
            value_counts.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f"Counts of {selected_column}", fontsize=16)
            ax.set_xlabel("Values", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            st.pyplot(fig)

        st.header("3️⃣ Box Plots for Numerical Columns (Outlier Detection)")
        st.write("Box Plot for Outlier Detection")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data[['Age', 'Weight (in Kg)', 'Height', 'PeriodDuration']])
        plt.title('Boxplot for Outlier Detection')
        st.pyplot(plt)

        st.header("4️⃣ Missing Values Heatmap")
        plt.figure(figsize=(12, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis', cbar_kws={'label': 'Missing Values'}, annot=False, xticklabels=True, yticklabels=False)
        plt.title('Missing Values Heatmap')
        st.pyplot(plt)

        st.header("5️⃣ PCOS Analysis Based on Feature (Grouped Aggregations)")
        selected_column = st.selectbox("Select a column to analyze with PCOS", options=data.columns)

        if selected_column not in ['PCOS']: 
            pcos_selected_col_count = data.groupby(['PCOS', selected_column]).size().reset_index(name='count')
            pcos_selected_col_count['PCOS'] = pcos_selected_col_count['PCOS'].map({0: 'No PCOS', 1: 'PCOS'})

            fig = px.bar(pcos_selected_col_count, 
                         x='PCOS', y='count', color=selected_column, 
                         title=f'PCOS vs {selected_column} Relationship',
                         labels={'PCOS': 'PCOS Status', 'count': 'Count'},
                         barmode='group', 
                         color_discrete_map={data[selected_column].unique()[0]: 'blue', data[selected_column].unique()[1]: 'orange'})
            st.plotly_chart(fig)
        else:
            st.write("Please select a valid column other than 'PCOS'.")

        st.header("7️⃣ Pairwise Relationships")
        st.write("Scatter Matrix of Numerical Columns")
        sns.pairplot(data[numerical_columns])
        st.pyplot(plt)

        st.header("8️⃣ PCOS Analysis by Age Range")
        age_min, age_max = st.slider(
            "Select an age range:",
            int(data['Age'].min()),
            int(data['Age'].max()),
            (20, 30)  # Default range
        )
        filtered_data = data[(data['Age'] >= age_min) & (data['Age'] <= age_max)]
        pcos_count = filtered_data.groupby('PCOS').size().reset_index(name='Count')
        pcos_count['PCOS'] = pcos_count['PCOS'].map({0: 'No PCOS', 1: 'PCOS'})

     
        st.subheader(f"PCOS Status in Age Range {age_min} - {age_max}")
        fig = px.bar(
            pcos_count,
            x='PCOS',
            y='Count',
            color='PCOS',
            title=f"PCOS Status in Age Range {age_min} - {age_max}",
            labels={'PCOS': 'PCOS Status', 'Count': 'Number of Individuals'},
            color_discrete_map={'PCOS': 'orange', 'No PCOS': 'blue'}
        )
        st.plotly_chart(fig)

        st.header("9️⃣ Chi-Square Test for Relationship between PCOS and attributes")
   
        categorical_columns = ['MenstrualCycleRegularity', 'RecentWeightChange', 'Hirsutism', 
                       'SkinDarkening', 'HairLoss', 'Acne', 'FastFoodConsumption', 'ExerciseRegularity', 
                       'PCOS', 'MoodSwings']

        data.columns = data.columns.str.strip()

        selected_column = st.selectbox("Select a column to analyze with PCOS:", categorical_columns)


        if selected_column:
            
            contingency_table = pd.crosstab(data['PCOS'], data[selected_column])
            
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            
            st.write(f"Chi-Square Test for {selected_column} and PCOS:-")
            st.write(f"Chi-Square statistic: {chi2:.2f}")
            st.write(f"p-value: {p:.4f}")
            st.write(f"Degrees of Freedom: {dof}")
            
            # Interpretation based on p-value
            if p < 0.05:
                st.write(f"There is a significant relationship between PCOS and {selected_column}.")
            else:
                st.write(f"There is no significant relationship between PCOS and {selected_column}.")

       
        st.header("🔟 Dataset Information:")
        info_df = pd.DataFrame({
            'Non-Null Count': data.notnull().sum(),
            'Data Type': data.dtypes
        })
        st.table(info_df)

    if selected_option == "Model Training and Evaluation":
        st.header("Model Selection: Model, Predictions, and Evaluation Results")
        st.subheader("1️⃣ Encoding Variables")

        st.write("""
        Since my columns contain only 0 and 1 as entries, 
        there is no need to encode them further. These values are already in a 
        numerical format representing binary categorical variables, and Random
        Forest models can handle them directly. Since Random Forest can work with 
        binary variables as distinct categories, there is no need for additional encoding 
        or transformations such as one-hot encoding.
        """)


        st.subheader("2️⃣ Feature Scaling (Optional for Random Forest)")

        st.write("""
        We can scale the features using `StandardScaler` for numerical columns such as Age, Weight, etc.
        However, Random Forest models are not sensitive to the scale of features, 
        so scaling is not strictly necessary for this model. Nonetheless, it's good practice for some models.
        """)

        st.write("""
        We are not applying scaling here, as Random Forest can work directly with the raw data.
        """)

        st.subheader("3️⃣ Train-Test Split")

        st.write("""
        We will split the dataset into training (80%) and testing (20%) sets. 
        This allows us to evaluate the model's performance on unseen data.
        """)

        X = data.drop('PCOS', axis=1)  
        y = data['PCOS'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write(f"Training features shape: {X_train.shape}")
        st.write(f"Testing features shape: {X_test.shape}")
        st.write(f"Training target shape: {y_train.shape}")
        st.write(f"Testing target shape: {y_test.shape}")
        st.write(f"Total data shape: {data.shape}")

        st.subheader("4️⃣ Model Training: Random Forest Classifier")

        st.write("""
        Now, we will train a Random Forest model to predict PCOS using the available features.
        Random Forest is an ensemble method that works well with binary classification tasks like this.
        """)

    
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)

        st.subheader("5️⃣ Feature Importance")

        st.write("""
        Random Forest models provide feature importances based on how well each feature contributes to making the decision.
        We will visualize these importances to understand which features play the most significant role in predicting PCOS.
        """)

        importances = rf.feature_importances_
        sorted_idx = importances.argsort()
        plt.figure(figsize=(5,3))
        plt.barh(range(len(importances)), importances[sorted_idx], align='center')
        plt.yticks(range(len(importances)), X.columns[sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance for PCOS Prediction")
        st.pyplot(plt)

    
        st.subheader("6️⃣ Recursive Feature Elimination (RFE)")

        st.write("""
        Now, we will use Recursive Feature Elimination (RFE) to select the top 5 most important features.
        RFE works by recursively removing the least important features based on model performance.
        """)

        # RFE (Recursive Feature Elimination)
        rfe = RFE(estimator=rf, n_features_to_select=5)  
        rfe.fit(X_train, y_train)


        selected_features = X.columns[rfe.support_]
        st.write("Selected Features:", selected_features)
        st.subheader("7️⃣ Retraining on Selected Features")

        st.write("""
        We will now retrain the Random Forest model using only the top 5 selected features.
        """)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]


        rf.fit(X_train_selected, y_train)
        st.subheader("8️⃣ Accuracy Results")
        accuracy = rf.score(X_test_selected, y_test)
        st.write(f"Model Accuracy on Selected Features: {accuracy:.4f}")

    if selected_option == "Prediction on New Data":
        X = data.drop('PCOS', axis=1)  
        y = data['PCOS'] 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_

        sorted_idx = importances.argsort()
        rfe = RFE(estimator=rf, n_features_to_select=5)  
        rfe.fit(X_train, y_train)
        selected_features = X.columns[rfe.support_]
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        rf.fit(X_train_selected, y_train)
        st.header("🔮 Prediction on New Data: PCOS Prediction")
        st.write("Enter values for the top 5 selected features to predict if a person has PCOS:")

        input_data = {
            'Age': st.number_input('Age', min_value=0, max_value=100, value=25),
            'Weight (in Kg)': st.number_input('Weight (in Kg)', min_value=30.0, max_value=200.0, value=60.0),
            'Height': st.number_input('Height (in cm)', min_value=100, max_value=250, value=160),
            'MenstrualCycleRegularity': st.selectbox('MenstrualCycleRegularity', options=[1,2,3], index=0),
            'PeriodDuration': st.number_input('Period Duration (days)', min_value=1, max_value=20, value=5)
        }
        input_df = pd.DataFrame([input_data])
        prediction = rf.predict(input_df)  

        if prediction[0] == 1:
            st.write("Based on the input data, the person is predicted to have PCOS.")
        else:
            st.write("Based on the input data, the person is predicted not to have PCOS.")
    if selected_option == "Conclusion and Insights":
        st.header("📊 Conclusion and Insights on PCOS Prediction")

        st.subheader("Summary of the Analysis")
        st.write("""
            In this analysis, we explored various factors that contribute to the prediction of PCOS. 
            We built a Random Forest classifier model to predict whether a person has PCOS based on their age, weight, height, menstrual cycle regularity, and period duration.
        """)

        st.subheader("Model Performance")
        st.write("""
            The model achieved an accuracy of 89%, meaning it correctly predicted the presence or absence of PCOS in 85% of the cases. 
            This indicates that the model is reasonably effective at predicting the condition based on the selected features.
        """)
        st.subheader("Key Features for PCOS Prediction")
        st.write("""
            The most important features for predicting PCOS were:
            - Age
            - Weight (in Kg)
            - Menstrual Cycle Regularity
            - Period Duration
            The model relied heavily on these features to make predictions.
        """)
        st.subheader("Actionable Insights")
        st.write("""
            - Women with irregular menstrual cycles and higher weight are at a higher risk of developing PCOS.
            - Monitoring age, weight, and menstrual cycle regularity can help in early detection of PCOS.
        """)
        st.subheader("Recommendations")
        st.write("""
            - It is recommended to track lifestyle factors, such as diet and exercise, in addition to the key features used in this analysis.
            - If you are experiencing symptoms of PCOS, we recommend consulting a healthcare professional for personalized advice and treatment.
        """)
        st.subheader("Future Work and Limitations")
        st.write("""
            - Further data collection from diverse populations could improve the model's accuracy and generalization.
            - The current model is based on only a few features, and incorporating additional factors such as genetics and lifestyle could improve prediction accuracy.
        """)

st.markdown(
    """
    <hr style='border: 1px solid #e0e0e0;'>
    <footer class='footer'>
        <p> 🌞 PCOS Prediction and Analysis App - Created by Areeba Chaudhry</p>
    </footer>
    """,
    unsafe_allow_html=True
)

            
