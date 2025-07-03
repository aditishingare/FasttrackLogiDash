
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(layout="wide")
st.title("FastTrack Logistics Feasibility Dashboard")

# Load data
df = pd.read_csv("fasttrack_synthetic_data.csv")

# Sidebar
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Data Visualisation", "Classification", "Clustering", "Association Rule Mining", "Regression"])

# Tab 1: Data Visualisation
if tab == "Data Visualisation":
    st.header("Descriptive Insights")
    st.write("Summary statistics and trends in the logistics dataset.")
    st.dataframe(df.head())

    fig1, ax1 = plt.subplots()
    sns.histplot(df["cur_delivery_time_hr"], kde=True, bins=30, ax=ax1)
    ax1.set_title("Delivery Time Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(x="industry", y="cur_cost_aed", data=df, ax=ax2)
    ax2.set_title("Cost per Parcel by Industry")
    st.pyplot(fig2)

    st.bar_chart(df["origin_city"].value_counts())

    st.line_chart(df.groupby("roi_months")["cur_cost_aed"].mean())

# Tab 2: Classification
elif tab == "Classification":
    st.header("Classification Models")
    st.write("Classify switch likelihood using various models.")

    features = ['cur_delivery_time_hr', 'cur_cost_aed', 'fuel_cost', 'driver_wage']
    data = df[features + ['switch_likelihood']].dropna()
    X = data[features]
    y = data['switch_likelihood']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_choice = st.selectbox("Select Classifier", ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"])
    if model_choice == "KNN":
        model = KNeighborsClassifier()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    if st.checkbox("Show Confusion Matrix"):
        st.write(confusion_matrix(y_test, y_pred))

# Tab 3: Clustering
elif tab == "Clustering":
    st.header("Customer Segmentation using K-Means")
    st.write("Segment customers using clustering on delivery metrics.")

    k = st.slider("Select number of clusters", 2, 10, 3)
    cluster_features = ['cur_cost_aed', 'avg_distance_km', 'pct_same_day']
    cluster_data = df[cluster_features].dropna()

    if len(cluster_data) >= k:
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['cluster'] = kmeans.fit_predict(cluster_data)
        st.dataframe(df[['industry', 'origin_city', 'cur_cost_aed', 'cluster']].head())

        fig, ax = plt.subplots()
        sns.scatterplot(x="avg_distance_km", y="cur_cost_aed", hue="cluster", data=df, palette="tab10", ax=ax)
        st.pyplot(fig)

        st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")
    else:
        st.warning("Not enough samples for the selected number of clusters.")

# Tab 4: Association Rule Mining
elif tab == "Association Rule Mining":
    st.header("Association Rule Mining on Adoption Drivers")
    st.write("Discover frequent feature combinations.")

    transactions = df["adoption_drivers"].dropna().str.split(", ").tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_ary, columns=te.columns_)

    min_support = st.slider("Min Support", 0.01, 0.5, 0.05)
    min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5)
    freq_items = apriori(trans_df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)

    top10_rules = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(top10_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Tab 5: Regression
elif tab == "Regression":
    st.header("Cost Modeling with Regression")
    st.write("Use regression models to estimate delivery cost.")

    reg_features = ['avg_distance_km', 'fuel_cost', 'driver_wage', 'maint_cost_per_km']
    data = df[reg_features + ['cur_cost_aed']].dropna()
    X = data[reg_features]
    y = data['cur_cost_aed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        st.write(f"{name} R² Score: {score:.2f}")
