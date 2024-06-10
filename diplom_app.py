import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

# Определение модели
class StudentPerformanceNN(nn.Module):
    def __init__(self):
        super(StudentPerformanceNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Инициализация модели и загрузка весов
model = StudentPerformanceNN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Загрузка скейлера
scaler = joblib.load('scaler.pkl')


# Страница для студентов
def student_page():
    st.title("Prediction of the risk zone for students")

    st.write("Enter your data to predict the zone of risk:")

    course_absence_rate = st.number_input("Course Absense Rate", min_value=0, max_value=100, value=0)
    pf = st.number_input("Pre Final score", min_value=0, max_value=100, value=0)

    if st.button("Predict"):
        # Нормализация данных с использованием StandardScaler
        X = np.array([[course_absence_rate, pf]])
        X_scaled = scaler.transform(X)

        # Прогнозирование
        inputs = torch.tensor(X_scaled, dtype=torch.float32)
        prediction = model(inputs)
        predicted_class = torch.argmax(prediction, dim=1).item()

        st.write(f"Predicted risk level class: ", predicted_class)


# Страница для учителей
def teacher_page():
    st.title("Predicting the level of risk for teachers")

    st.write("Upload a file of student data to predict the level of risk:")

    uploaded_file = st.file_uploader("Select a file", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("Uploaded data:")
        st.write(data.head())

        if st.button("Predict"):
            # Предполагается, что данные содержат столбцы "Course Absence Rate" и "PF"
            X = data[['Course Absence Rate', 'PF']].values / 100.0

            # Прогнозирование
            inputs = torch.tensor(X, dtype=torch.float32)
            predictions = model(inputs).detach().numpy()
            predicted_classes = np.argmax(predictions, axis=1)
            data['FIN Prediction'] = predicted_classes
            st.write("Данные с предсказанными оценками (FIN):")
            st.write(data)

            # Скачивание результата
            csv = data.to_csv(index=False)
            st.download_button(label="Download result",
                               data=csv,
                               file_name='predictions.csv',
                               mime='text/csv')


# Выбор страницы
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to page:", ["Student", "Teacher"])

if page == "Student":
    student_page()
else:
    teacher_page()
