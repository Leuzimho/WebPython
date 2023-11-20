from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_test', methods=['POST'])
def train_test():
    df = pd.read_csv("auto-mpg.csv")

    selected_columns = ['weight', 'mpg']
    X = df[selected_columns]
    y = df['cylinders']  

    # divisao dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    classifier_name = request.form['classifier']
    if classifier_name == 'KNN':
        # número de vizinhos do formulário ou um valor padrão (5)
        k_neighbors = int(request.form.get('k_neighbors', 5))
        classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
    elif classifier_name == 'SVM':
        # valor C do formulário ou um valor padrão (1.0)
        svm_c = float(request.form.get('svm_c', 1.0))
        classifier = SVC(C=svm_c, kernel='linear')
    elif classifier_name == 'MLP':
        # tamanho da camada oculta do formulário ou um valor padrão (100,)
        mlp_hidden_layer_sizes = tuple(map(int, request.form.get('mlp_hidden_layer_sizes', '100').split(',')))
        classifier = MLPClassifier(hidden_layer_sizes=mlp_hidden_layer_sizes, max_iter=1000)
    elif classifier_name == 'DT':
        classifier = DecisionTreeClassifier()
    elif classifier_name == 'RF':
        classifier = RandomForestClassifier(n_estimators=100)

    # treinamento do modelo
    classifier.fit(X_train, y_train)


    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = set(y_test)
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

    plt.tight_layout()


    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('result.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
