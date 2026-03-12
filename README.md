# Healthcare-Bias-Detection-T-Net-

A machine learning based web application that detects potential bias in healthcare clinical data using a **T-Net deep learning architecture**. The system allows users to analyze datasets and visualize bias patterns through an interactive web interface built with **Streamlit**.

This project highlights **algorithmic bias in healthcare datasets**, helping developers and researchers build more fair and reliable AI systems.

---

## 🚀 Features

- 🔐 **User Authentication**
  - Login and Signup functionality
  - Password hashing for security
  - SQLite database storage

- 🤖 **Bias Detection Model**
  - Custom **T-Net neural network architecture**
  - Built using **PyTorch**

- 📊 **Interactive Data Visualization**
  - Bias analysis charts using **Plotly**
  - Feature comparison and dataset insights

- 🌐 **Web Interface**
  - Developed using **Streamlit**
  - Easy-to-use dashboard for bias detection

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|--------|
| Python | Programming Language |
| PyTorch | Deep Learning Model |
| Streamlit | Web Application |
| Plotly | Data Visualization |
| Pandas | Data Processing |
| SQLite | User Authentication Database |

---

## 📂 Project Structure

```
Healthcare-Bias-Detection-T-Net/
│
├── app.py
├── tnet_model.py
├── generate_dataset.py
├── clinical_dataset_biased.csv
├── users.db
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Healthcare-Bias-Detection-T-Net.git
cd Healthcare-Bias-Detection-T-Net
```

---

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Mac / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser and go to:

```
http://localhost:8501
```

---

## 📊 Dataset

The project includes a sample dataset:

```
clinical_dataset_biased.csv
```

This dataset simulates healthcare records containing potential bias patterns for testing and research.

You can also generate a new dataset using:

```bash
python generate_dataset.py
```

---

## 🧠 Model

The system uses a **T-Net (Transformation Network)** implemented in **PyTorch** to analyze healthcare data and detect bias patterns such as:

- demographic disparities
- treatment differences
- unfair predictions

---

## 📈 Visualization

The application provides interactive charts for:

- Bias distribution
- Feature comparisons
- Dataset insights

These visualizations help users easily understand bias present in the data.

---

## 🚀 Future Improvements

- Integration with real-world healthcare datasets
- Explainable AI (XAI) for bias interpretation
- Deployment using Docker or cloud platforms
- Advanced fairness metrics
- Role-based user access system
