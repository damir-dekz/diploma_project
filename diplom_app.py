import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# Определение модели
class StudentPerformanceNN(nn.Module):
    def __init__(self):
        super(StudentPerformanceNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Инициализация модели
model = StudentPerformanceNN()
model.load_state_dict(torch.load('model.pth'))  # Загрузите сохраненную модель
model.eval()


# Страница для студентов
def student_page():
    st.title("Прогноз оценки для студентов")

    st.write("Введите ваши данные для прогнозирования итоговой оценки (FIN):")

    course_absence_rate = st.number_input("Процент пропуска занятий", min_value=0, max_value=100, value=0)
    pf = st.number_input("Предварительные итоговые баллы(в процентах) (PF)", min_value=0, max_value=100, value=0)

    if st.button("Предсказать"):
        # Нормализация данных
        course_absence_rate = course_absence_rate/100
        pf = pf/100
        X = np.array([[course_absence_rate, pf]])

        # Прогнозирование
        inputs = torch.tensor(X, dtype=torch.float32)
        prediction = model(inputs).item()

        st.write(f"Предсказанная итоговая оценка(в процентах) (FIN): {round(prediction*100)}")


# Страница для учителей
def teacher_page():
    st.title("Прогнозирование оценок для учителей")

    st.write("Загрузите файл с данными студентов для прогнозирования итоговых оценок (FIN):")

    uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("Загруженные данные:")
        st.write(data.head())

        if st.button("Предсказать"):
            # Предполагается, что данные содержат столбцы "Course Absence Rate" и "PF"
            X = data[['Course Absence Rate', 'PF']].values

            # Прогнозирование
            inputs = torch.tensor(X, dtype=torch.float32)
            predictions = model(inputs).detach().numpy()
            for i in range(len(predictions)):
                predictions[i] = np.round(predictions[i],2) if predictions[i] > 0 else 0
            data['FIN Prediction'] = predictions
            st.write("Данные с предсказанными оценками (FIN):")
            st.write(data)

            # Скачивание результата
            csv = data.to_csv(index=False)
            st.download_button(label="Скачать результат",
                               data=csv,
                               file_name='predictions.csv',
                               mime='text/csv')


# Выбор страницы
st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти на страницу:", ["Студенты", "Учителя"])

if page == "Студенты":
    student_page()
else:
    teacher_page()
