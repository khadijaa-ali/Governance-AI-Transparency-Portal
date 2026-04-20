import pandas as pd
import numpy as np
import random 
np.random.seed(42)
random.seed(42)
sample =1600

# dataset

department_list = [
    "Health","Education","Transport","Energy","Water & Sanitation","Housing & Works","Agriculture","Defense","Finance","Interior"
]
company_mapping = {
    "Health": ["MediCore Supplies", "HealthCare Plus"],
    "Education": ["EduSmart Innovations", "NextGen Learning"],
    "Transport": ["UrbanMove Systems", "MetroBuild Constructors"],
    "Energy": ["EcoEnergy Systems", "SolarNova Power"],
    "Water & Sanitation": ["AquaWorks International", "BlueWave Technologies"],
    "Housing & Works": ["GreenLiving Estates", "BuildSmart Solutions"],
    "Agriculture": ["AgriNova Supplies", "FarmTech Solutions"],
    "Defense": ["ShieldSecure Systems", "ArmorTech Solutions"],
    "Finance": ["FinEdge Analytics", "SecureFunds Corp"],
    "Interior": ["SafeCity Solutions", "Guardian Tech"]
}
amount_ranges = {

    "Health": (400000, 900000),
    "Education": (300000, 700000),
    "Transport": (800000, 2000000),
    "Energy": (1000000, 5000000),
    "Water & Sanitation": (500000, 1500000),
    "Housing & Works": (1000000, 3000000),
    "Agriculture": (300000, 800000),
    "Defense": (2000000, 8000000),
    "Finance": (200000, 600000),
    "Interior": (500000, 1200000)
}

department=np.random.choice(department_list , sample)
company = [random.choice(company_mapping[dept]) for dept in department]
amounts = [random.randint(amount_ranges[dept][0], amount_ranges[dept][1]) for dept in department]
date = pd.to_datetime(np.random.choice(pd.date_range(start="2021-01-01", end="2025-01-01"), sample ,replace=True))
status_option = ["Complete", "Ongoing", "Delayed"]
status = np.random.choice(status_option, size=sample, p=[0.6, 0.25, 0.15])
#  Randomly selecting 50 projects to have extreme amounts
for _ in range(50):
    idx = random.randint(0, sample-1)
    dept = department[idx]
    min_value, max_value = amount_ranges[dept]
    amount = random.choice([
        random.randint(max_value*2, max_value*5),  # high amount
        random.randint(max(1, min_value//10), min_value//2)  # low amount
    ])
    amounts[idx] = amount

#  Randomly assigning wrong seller for 100 projects
for _ in range(100):
    idx = random.randint(0, sample-1)
    wrong_dept = random.choice([d for d in department_list if d != department[idx]])
    company[idx] = random.choice(company_mapping[wrong_dept])

#  Randomly marking 50 high-budget projects as delayed
count = 0
while count < 50:
    idx = random.randint(0, sample-1)
    if amounts[idx] > amount_ranges[department[idx]][1]*1.5:
        status[idx] = "Delayed"
        count += 1

df = pd.DataFrame({
    "Department": department,
    "Company": company,
    "Amount": amounts,
    "Date": date,
    "Status": status
})

latest_date = df["Date"].max()
is_anomaly = []
for i in range(sample):
    department_name = df.loc[i, "Department"]
    company_name = df.loc[i, "Company"]
    project_amount = df.loc[i, "Amount"]
    project_status = df.loc[i, "Status"]
    project_date = df.loc[i, "Date"]
    min_value, max_value = amount_ranges[department_name]

    amount_anomaly = (project_amount > max_value * 1.5) or (project_amount < min_value * 0.5)
    company_anomaly = company_name not in company_mapping[department_name]
    status_anomaly = (project_amount > max_value * 1.5 and project_status == "Complete") or \
                     (project_amount < min_value * 0.5 and project_status == "Delayed")

    date_anomaly = (project_status == "Ongoing" and (latest_date - project_date).days > 400) or \
                   (project_status == "Complete" and project_date > latest_date) or \
                   (project_status == "Complete" and (latest_date - project_date).days < 30) or \
                   (project_status == "Delayed" and (latest_date - project_date).days > 600)

    if amount_anomaly or company_anomaly or status_anomaly or date_anomaly:
        is_anomaly.append(1)
    else:
        is_anomaly.append(0)

df["is_anomaly"] = is_anomaly
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

# feauture engineering

# Extract Year, Month, Day
df["Year"] = pd.to_datetime(df["Date"]).dt.year
df["Month"] = pd.to_datetime(df["Date"]).dt.month
df["Day"] = pd.to_datetime(df["Date"]).dt.day

#  Project Size Category
def categorize_amount(x):
    if x < 500000:
        return "Small"
    elif x < 2000000:
        return "Medium"
    else:
        return "Large"
df["Project_Size"] = df["Amount"].apply(categorize_amount)

#  Days Since Latest Project
df["Days_Since_Latest"] = (pd.to_datetime(df["Date"]).max() - pd.to_datetime(df["Date"])).dt.days

# Department-Company Match
df["Dept_Company_Match"] = df.apply(
    lambda row: 1 if row["Company"] in company_mapping[row["Department"]] else 0, axis=1
)

# Log Amount
df["Log_Amount"] = np.log1p(df["Amount"])


df.to_csv("gov_Dataset.csv", index=False)
print("Dataset Created")

# checking data

# df.info()
# df.describe() 
# df.head()
# df['Department'].value_counts()

# ML part
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# One-hot encoding
df_enc = pd.get_dummies(df, columns=["Department", "Company", "Status", "Project_Size"], drop_first=True)
X = df_enc.drop(columns=["is_anomaly", "Date"])
y = df_enc["is_anomaly"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
}

best_model = None
best_score = -1
best_name = ""

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    roc = roc_auc_score(y_test, predictions)

    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC-AUC: {roc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 40)

   
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {model_name}")
    plt.show()

    # checking best model 
    if roc > best_score:
        best_score = roc
        best_model = model
        best_name = model_name


joblib.dump(best_model, "anomaly_detector.pkl")
print(f"Best model saved as anomaly_detector.pkl ({best_name} with ROC-AUC {best_score:.3f})")


# data visualization


plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Department', hue='is_anomaly', palette={0:'blue', 1:'red'})
plt.xticks(rotation=45)
plt.title("Projects per Department (Red = Anomalies)")
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data=df, x='Amount', hue='is_anomaly', bins=50, palette={0:'blue', 1:'red'}, multiple="stack")
plt.title("Distribution of Project Amounts (Red = Anomalies)")
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(data=df, x='Status', hue='is_anomaly', palette={0:'green', 1:'orange'})
plt.title("Project Status with Anomalies Highlighted")
plt.show()

plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x='Date', y='Amount', hue='is_anomaly', palette={0:'blue', 1:'red'}, alpha=0.6)
plt.title("Project Amounts Over Time (Red = Anomalies)")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,6))
anomaly_year = df.groupby("Year")["is_anomaly"].mean()
sns.barplot(x=anomaly_year.index, y=anomaly_year.values, color="red")
plt.title("Anomaly Ratio per Year")
plt.ylabel("Ratio of Anomalies")
plt.show()

# Random Forest important feautures
if isinstance(best_model, RandomForestClassifier):
    importances = best_model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:15]

    plt.figure(figsize=(10,6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
    plt.title("Top 15 Important Features (Random Forest)")
    plt.show()


  frontend

  import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest
import numpy as np
import re 

cnic_df = pd.read_csv(r"C:\Users\hp\Desktop\python\cnic_dataset.csv")
bill_df = pd.read_csv(r"C:\Users\hp\Desktop\python\bill_dataset.csv")
gov_df = pd.read_csv(r"C:\Users\hp\Desktop\python\gov_Dataset.csv")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Chatbot", "Dashboard", "Anomaly Detection"])

# chatbot
if menu == "Chatbot":
    st.title("Citizen Service Chatbot")

    lang = st.radio("Select Language / زبان منتخب کریں", ["English", "اردو"])

    def validate_cnic(cnic):
        pattern = r"^\d{5}-?\d{7}-?\d{1}$"  
        return re.match(pattern, cnic) is not None

    if lang == "English":
        st.header("Welcome. Please select a service")
        service = st.selectbox("Choose a service", ["Utility Bill Inquiry", "CNIC Verification"])
    
        if service == "Utility Bill Inquiry":
            ref = st.text_input("Enter your Reference Number")
            if st.button("Check Bill"):
                record = bill_df[bill_df["Reference_No"] == ref]
                if not record.empty:
                    st.success(f"CNIC: {record.iloc[0]['CNIC']}, Bill: {record.iloc[0]['Bill_Amount']} PKR, Due: {record.iloc[0]['Due_Date']}, Status: {record.iloc[0]['Status']}")
                else:
                    st.error("Reference number not found")
    
        if service == "CNIC Verification":
            cnic = st.text_input("Enter your CNIC")
            if st.button("Verify CNIC"):
                if validate_cnic(cnic):
                    record = cnic_df[cnic_df["CNIC"] == cnic]
                    if not record.empty:
                        st.success(f"Name: {record.iloc[0]['Name']}, DOB: {record.iloc[0]['DOB']}, Status: {record.iloc[0]['Status']}")
                    else:
                        st.error("CNIC not found in records")
                else:
                    st.error("Invalid CNIC format. Please enter 13 digits only.")

    elif lang == "اردو":
        st.header("خوش آمدید۔ براہ کرم سروس منتخب کریں")
        service = st.selectbox("سروس منتخب کریں", ["بل انکوائری", "شناختی کارڈ ویریفکیشن"])
    
        if service == "بل انکوائری":
            ref = st.text_input("اپنا ریفرنس نمبر درج کریں")
            if st.button("بل چیک کریں"):
                record = bill_df[bill_df["Reference_No"] == ref]
                if not record.empty:
                    st.success(f"شناختی کارڈ: {record.iloc[0]['CNIC']}, بل: {record.iloc[0]['Bill_Amount']} روپے, آخری تاریخ: {record.iloc[0]['Due_Date']}, حیثیت: {record.iloc[0]['Status']}")
                else:
                    st.error("ریفرنس نمبر نہیں ملا")
    
        if service == "شناختی کارڈ ویریفکیشن":
            cnic = st.text_input("اپنا شناختی کارڈ نمبر درج کریں")
            if st.button("ویریفائی کریں"):
                if validate_cnic(cnic):
                    record = cnic_df[cnic_df["CNIC"] == cnic]
                    if not record.empty:
                        st.success(f"نام: {record.iloc[0]['Name']}, تاریخ پیدائش: {record.iloc[0]['DOB']}, حیثیت: {record.iloc[0]['Status']}")
                    else:
                        st.error("شناختی کارڈ نہیں ملا")
                else:
                    st.error("غلط شناختی کارڈ نمبر۔ براہ کرم 13 ہندسوں کا نمبر درج کریں")


# dashboard
elif menu == "Dashboard":
    st.title("Government Budget Visualization Dashboard")
    st.sidebar.header("Filters")

    year_filter = st.sidebar.multiselect("Select Year", options=gov_df["Year"].unique(), default=gov_df["Year"].unique())
    dept_filter = st.sidebar.multiselect("Select Department", options=gov_df["Department"].unique(), default=gov_df["Department"].unique())

    df_filtered = gov_df[(gov_df["Year"].isin(year_filter)) & (gov_df["Department"].isin(dept_filter))]

    fig1 = px.bar(df_filtered.groupby("Department")["Amount"].sum().reset_index(),
                  x="Department", y="Amount", color="Department",
                  title="Total Budget Allocation by Department")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df_filtered, x="Date", y="Amount",
                      color=df_filtered["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
                      title="Project Amounts Over Time with Anomalies")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(df_filtered.groupby("Year")["Amount"].sum().reset_index(),
                   x="Year", y="Amount", title="Yearly Spending Trend")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.pie(df_filtered, names="Status", title="Project Status Distribution")
    st.plotly_chart(fig4, use_container_width=True)

    anomaly_ratio = df_filtered.groupby("Department")["is_anomaly"].mean().reset_index()
    fig5 = px.bar(anomaly_ratio, x="Department", y="is_anomaly", title="Anomaly Ratio by Department")
    st.plotly_chart(fig5, use_container_width=True)

elif menu == "Anomaly Detection":
    st.title("AI-based Anomaly Detection in Procurement Data")

    # Use Amount column for anomaly detection
    model = IsolationForest(contamination=0.1, random_state=42)
    gov_df["anomaly_pred"] = model.fit_predict(gov_df[["Amount"]])

    gov_df["anomaly_status"] = gov_df["anomaly_pred"].map({1: "Normal", -1: "Anomaly"})

    st.write("Sample of detected anomalies:")
    st.dataframe(gov_df[["Department", "Amount", "Status", "anomaly_status"]].head(15))

    # Visualize anomalies
    fig = px.scatter(gov_df, x="Date", y="Amount",
                     color="anomaly_status",
                     title="Anomaly Detection Results on Procurement Data")
    st.plotly_chart(fig, use_container_width=True)





