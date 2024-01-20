from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Tworzenie macierzy pomylek
def create_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predykowane Etykiety')
    plt.ylabel('Prawdziwe Etykiety')
    plt.title('Macierz Pomyłek')
    plt.show()


# Tworzenie krzywej ROC
def create_roc(true_labels, predicted_probabilities):
    fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
    roc_auc = roc_auc_score(true_labels, predicted_probabilities)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend(loc="lower right")
    plt.show()


def read_data():
    column_names = ["ID", "Outcome", "Time"] + [f"feature_{i}" for i in range(4, 34)] + ["Tumor_Size", "Lymph_Node_Status"]
    data = pd.read_csv('data/wpbc.data', header=None, names=column_names)
    return data


def convert_question_mark_into_median(data):
    data["Lymph_Node_Status"] = pd.to_numeric(data["Lymph_Node_Status"], errors='coerce')
    imputer = SimpleImputer(strategy="median")
    data["Lymph_Node_Status"] = imputer.fit_transform(data[["Lymph_Node_Status"]])
    return data


def main():
    # Wczytanie danych
    data = read_data()
    data = convert_question_mark_into_median(data)

    # Konwersja Outcome do postaci binarnej (R = 1, N = 0)
    data["Outcome"] = LabelEncoder().fit_transform(data["Outcome"])

    # Podział danych na cechy (X) i etykiety (y)
    X = data.drop(columns=["ID", "Outcome"])
    y = data["Outcome"]

    # Normalizacja danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=85)

    # Stworzenie klasyfikatora Bayesowskiego i trenowanie go, następnie predykcja
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)

    # Accuracy klasyfikatora
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Raport klasyfikacji
    print(classification_report(y_test, y_pred))

    # Dokonanie predykcji prawdopodobieństw dla klasy 1
    predicted_probabilities_1 = nb_classifier.predict_proba(X_test)[:, 1]
    predicted_labels = (predicted_probabilities_1 >= 0.5).astype(int)

    # Zamiana etykiet na binarne (0 lub 1)
    true_labels = y_test.map(lambda x: 1 if x == 1 else 0)
    create_confusion_matrix(true_labels, predicted_labels)
    create_roc(true_labels, predicted_probabilities_1)


if __name__ == '__main__':
    main()

