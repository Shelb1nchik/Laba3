import os
from flask import Flask, request, render_template, redirect, url_for
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('models', exist_ok=True)

DATA_PATH = None
MODEL_PATH = 'models/model.pkl'
RESULTS = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global DATA_PATH
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        DATA_PATH = file_path
        return redirect(url_for('visualize'))
    return 'No file uploaded', 400

@app.route('/visualize')
def visualize():
    if not DATA_PATH:
        return 'No data available', 400
    df = dd.read_csv(DATA_PATH)
    sample = df.sample(frac=0.1).compute()
    num_df = sample.select_dtypes(include='number')
    if num_df.shape[1] < 2:
        return 'Not enough numeric columns for visualization', 400
    sns.pairplot(num_df)
    plt.savefig('static/plot.png')
    plt.close()
    return render_template('visualize.html', image_path='static/plot.png')

@app.route('/train', methods=['POST'])
def train():
    global RESULTS
    if not DATA_PATH:
        return 'No data available', 400
    target = request.form['target']
    df = dd.read_csv(DATA_PATH).compute()
    if target not in df.columns:
        return 'Invalid target column', 400
    X = df.drop(columns=[target])
    y = df[target]
    X = X.select_dtypes(include='number')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    y_pred = model.predict(X_test)

    full_report = classification_report(y_test, y_pred, output_dict=True)
    # фильтруем только строки с числовыми метриками
    RESULTS['report'] = {
        label: metrics for label, metrics in full_report.items()
        if isinstance(metrics, dict) and 'precision' in metrics
    }

    matrix = confusion_matrix(y_test, y_pred)
    RESULTS['matrix'] = matrix.tolist()

    # Сохраняем визуализацию матрицы ошибок
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('static/conf_matrix.png')
    plt.close()

    return redirect(url_for('results'))

@app.route('/results')
def results():
    return render_template('results.html', report=RESULTS.get('report'), matrix=RESULTS.get('matrix'), image_path='static/conf_matrix.png')

#starting
if __name__ == '__main__':
    app.run(debug=True)