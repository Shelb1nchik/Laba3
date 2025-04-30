Установка зависимостей

Выполните:
pip install -r requirements.txt

Список нужных библиотек:
Flask  
dask[dataframe]  
matplotlib  
seaborn  
scikit-learn  
joblib

Запуск приложения:
python app.py
Приложение будет доступно по адресу: http://127.0.0.1:5000

Использование:
1. Загрузка данных
Перейдите на главную страницу.

Загрузите CSV-файл с числовыми признаками и колонкой target.

2. Визуализация
После загрузки вы увидите pairplot с примерами данных.

3. Обучение модели
Введите название колонки target, нажмите Обучить модель.

4. Просмотр результатов
Появятся метрики классификации в виде таблицы и матрица ошибок в виде графика.

Структура проекта
laba3/  
├── app.py  
├── requirements.txt  
├── README.md  
├── data/  
├── static/  
│   └── plot.png  
│   └── conf_matrix.png  
├── templates/  
│   ├── index.html  
│   ├── visualize.html  
│   └── results.html  
└── models/  
