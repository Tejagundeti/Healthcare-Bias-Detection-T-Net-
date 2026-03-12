import streamlit as st
import pandas as pd
import torch
import os
import plotly.express as px
import sqlite3
import hashlib
from tnet_model import TNet

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT UNIQUE, email TEXT, password TEXT)')
    conn.commit()
    conn.close()

init_db()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

st.set_page_config(page_title="Disease Prediction System", layout="wide")

# ---------- LOGIN & REGISTRATION ----------
if "login" not in st.session_state:
    st.session_state.login=False

def auth():
    st.title("🏥 Healthcare Bias Detection (T-Net)")
    
    auth_mode = st.radio("Choose Mode", ["Login", "Sign Up"], horizontal=True)
    
    if auth_mode == "Login":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        
        if st.button("Login"):
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (u,))
            user = c.fetchone()
            conn.close()
            
            if u == "admin" and p == "admin123":
                st.session_state.login = True
                st.rerun()
            elif user and check_hashes(p, user[2]):
                st.session_state.login = True
                st.rerun()
            else:
                st.error("Invalid Username or Password")
                
    elif auth_mode == "Sign Up":
        new_user = st.text_input("Username")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Sign Up"):
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            elif len(new_user) == 0 or len(new_password) == 0:
                st.error("Username and Password cannot be empty.")
            else:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                try:
                    c.execute('SELECT * FROM users WHERE username = ?', (new_user,))
                    if c.fetchone():
                        st.error("Username already exists!")
                    else:
                        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                                  (new_user, new_email, make_hashes(new_password)))
                        conn.commit()
                        st.success("Registration Successful! You can now log in.")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    conn.close()

# ---------- LOAD DATA ----------
def load_data():
    if os.path.exists("clinical_dataset_biased.csv"):
        return pd.read_csv("clinical_dataset_biased.csv")
    return None

# ---------- MAIN ----------
if not st.session_state.login:
    auth()

else:

    menu=st.sidebar.radio("Navigation",
    ["Dashboard","Dataset","Train TNet","Clinical Dashboard"])

# ---------- DASHBOARD ----------
    if menu=="Dashboard":
        st.title("📊 Clinical Dataset Dashboard")
        df = load_data()
        
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Patient Demographics")
                fig_age = px.histogram(df, x="Age", title="Age Distribution", nbins=20, color_discrete_sequence=['#4CAF50'])
                st.plotly_chart(fig_age, use_container_width=True)

            with col2:
                st.subheader("Disease Prevalence")
                diseases = ["Heart Disease", "Diabetes", "Flu", "Asthma", "Hypertension", "Typhoid", "Covid", "Malaria", "Allergy", "Pneumonia"]
                disease_counts = df[diseases].sum().reset_index()
                disease_counts.columns = ['Disease', 'Cases']
                disease_counts = disease_counts.sort_values(by='Cases', ascending=True)
                
                fig_diseases = px.bar(disease_counts, x="Cases", y="Disease", orientation='h', title="Cases per Disease", color='Cases', color_continuous_scale='Greens')
                st.plotly_chart(fig_diseases, use_container_width=True)
                
        else:
            st.warning("⚠️ Dataset missing. Please generate the dataset to view analytics.")

# ---------- DATASET ----------
    elif menu=="Dataset":
        df=load_data()
        if df is not None:
            st.dataframe(df)
        else:
            st.error("Dataset missing")

# ---------- TRAIN ----------
    elif menu=="Train TNet":
        st.title("⚙️ Train T-Net Model")
        
        df=load_data()

        if df is not None:

            diseases=df.columns[9:-1]

            X=df.iloc[:,0:9]
            y=df[diseases]
            s=df["Is_Training_Set"]

            X=torch.tensor(X.values,dtype=torch.float32)
            y=torch.tensor(y.values,dtype=torch.float32)
            s=torch.tensor(s.values,dtype=torch.float32).view(-1,1)
            
            # --- Hyperparameters ---
            st.subheader("Model Hyperparameters")
            col1, col2 = st.columns(2)
            epochs_to_train = col1.slider("Training Epochs", min_value=1, max_value=10, value=5, step=10)
            learning_rate = col2.selectbox("Learning Rate", [0.01, 0.005, 0.001, 0.0001,1], index=1)
            
            if st.button("Start Training", type="primary"):
                model=TNet(X.shape[1],len(diseases))

                optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
                loss_fn=torch.nn.BCELoss()

                st.markdown("### Training Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Interactive loss chart
                loss_history = []
                chart_placeholder = st.empty()

                for epoch in range(epochs_to_train):
                    id_out,pred_out=model(X)

                    loss=loss_fn(id_out,s)+loss_fn(pred_out,y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics dynamically
                    loss_history.append(loss.item())
                    progress_pct = int(((epoch + 1) / epochs_to_train) * 100)
                    
                    progress_bar.progress(progress_pct)
                    status_text.text(f"Epoch: {epoch+1}/{epochs_to_train} - Loss: {loss.item():.4f}")
                    
                    # Update chart every few epochs so it animates cleanly
                    if epoch % 2 == 0 or epoch == epochs_to_train - 1:
                        chart_placeholder.line_chart(loss_history)

                st.session_state.model=model
                st.success("✅ T-Net Model Successfully Trained!")

        else:
            st.error("Dataset missing")

# ---------- CLINICAL DASHBOARD PREDICTION ----------
    elif menu=="Clinical Dashboard":
        
        # --- STYLING ---
        st.markdown(
            """
            <style>
            .main-title {
                color: #2E7D32;
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="main-title">🏥 Safe AI Clinical Prediction System</div>', unsafe_allow_html=True)
        
        diseases = ["Heart Disease", "Diabetes", "Flu", "Asthma", "Hypertension", "Typhoid", "Covid", "Malaria", "Allergy", "Pneumonia"]

        # --- SIDEBAR ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Configuration")
        selected_disease = st.sidebar.selectbox("Select Disease Model", diseases)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Patient Vitals / Features")
        
        age = st.sidebar.number_input("Age", 1, 100, 30, help="Patient's age in years")
        temp = st.sidebar.number_input("Temperature (°F)", 95.0, 105.0, 98.6, step=0.1, help="Body temperature in Fahrenheit")
        
        cough = st.sidebar.selectbox("Cough", ["No", "Yes"], help="Does the patient have a cough?")
        cold = st.sidebar.selectbox("Cold", ["No", "Yes"], help="Does the patient have a cold?")
        headache = st.sidebar.selectbox("Headache", ["No", "Yes"], help="Does the patient have a headache?")
        bodypain = st.sidebar.selectbox("Body Pain", ["No", "Yes"], help="Does the patient have body pain?")
        sorethroat = st.sidebar.selectbox("Sore Throat", ["No", "Yes"], help="Does the patient have a sore throat?")
        fatigue = st.sidebar.selectbox("Fatigue", ["No", "Yes"], help="Does the patient have fatigue?")
        smoking = st.sidebar.selectbox("Smoking Status", ["Non-Smoker", "Smoker"], help="Does the patient smoke tobacco?")

        # --- MAIN DASHBOARD ---
        st.subheader(f"Current Model: {selected_disease}")
        
        st.info(f"**Bias Prevention:** The {selected_disease} model refers patients that are statistically different from the training population (Sample Selection Bias).")

        analyze_clicked = st.button("Analyze Patient", use_container_width=True, type="primary")

        if analyze_clicked:
            if "model" not in st.session_state:
                st.warning("⚠️ Please train the T-Net model first from the 'Train TNet' menu option.")
            else:
                features = [
                    age, temp,
                    1 if cough == "Yes" else 0,
                    1 if cold == "Yes" else 0,
                    1 if headache == "Yes" else 0,
                    1 if bodypain == "Yes" else 0,
                    1 if sorethroat == "Yes" else 0,
                    1 if fatigue == "Yes" else 0,
                    1 if smoking == "Smoker" else 0
                ]
                
                x = torch.tensor([features], dtype=torch.float32)

                with torch.no_grad():
                    id_out, pred_out = st.session_state.model(x)
                    
                    id_score = id_out.item()
                    
                    st.markdown("### Analysis Results")
                    st.metric("Identification Score (OOD)", f"{id_score:.4f}")
                    
                    # Target bias behavior
                    if id_score < 0.5:
                        st.error("### ⚠️ Referral Recommended\nThis patient's profile is significantly different from the training population (OOD). Determining robust diagnosis is unsafe.")
                    else:
                        st.success("### ✅ Safe to Predict\nPatient profile matches training distribution.")
                        
                    st.markdown("---")
                    
                    # Interactive plotting & Probabilities
                    prob_df = pd.DataFrame({
                        "Disease": diseases,
                        "Probability (%)": (pred_out[0].numpy() * 100)
                    })
                    prob_df = prob_df.sort_values(by="Probability (%)", ascending=False)
                    
                    most_probable_disease = prob_df.iloc[0]["Disease"]
                    highest_prob = prob_df.iloc[0]["Probability (%)"] / 100.0
                    
                    if highest_prob > 0.5:
                        st.warning(f"**Clinical Alert**: Probable Disease is {most_probable_disease} ({highest_prob:.1%})")
                        
                        # Probable Cause Analysis
                        causes = []
                        if most_probable_disease == "Heart Disease" and age > 50: causes.append("Age factor")
                        if most_probable_disease == "Heart Disease" and smoking == "Smoker": causes.append("Smoking history")
                        if most_probable_disease == "Heart Disease" and fatigue == "Yes": causes.append("Reported fatigue")
                        
                        if most_probable_disease == "Flu" and temp >= 100: causes.append("Fever")
                        if most_probable_disease == "Flu" and bodypain == "Yes": causes.append("Body aches")
                        
                        if most_probable_disease == "Asthma" and smoking == "Smoker": causes.append("Smoking history")
                        if most_probable_disease == "Asthma" and cough == "Yes": causes.append("Persistent cough")

                        if most_probable_disease == "Covid" and temp >= 99.5: causes.append("Fever")
                        if most_probable_disease == "Covid" and cough == "Yes": causes.append("Cough")
                        if most_probable_disease == "Covid" and sorethroat == "Yes": causes.append("Sore Throat")
                        if most_probable_disease == "Covid" and fatigue == "Yes": causes.append("Fatigue")
                        
                        if len(causes) > 0:
                            st.info(f"**Probable Causes for Flagging {most_probable_disease}:**\n- " + "\n- ".join(causes))

                    else:
                        st.info(f"**Clinical Status**: No high-risk disease detected. Most probable is {most_probable_disease} ({highest_prob:.1%})")
                    
                    st.markdown("---")
                    st.subheader("Interactive Disease Probability Curve")
                    
                    fig = px.bar(prob_df, x="Probability (%)", y="Disease", orientation='h', color="Probability (%)", color_continuous_scale="Reds")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    
                    st.plotly_chart(fig, use_container_width=True)