import random
from collections import Counter
from statistics import mode
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    return precision, recall, f1
    
def compute_metricsFullDetails(y_true, y_pred):
    # Υπολογισμός των μετρικών ακρίβειας (precision), ανάκλησης (recall) και F1 για καθε κατηγορια ξεχωριστα
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Υπολογισμός των μέσων όρων (micro και macro)
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return {
        "Micro Precision": micro_precision,
        "Micro Recall": micro_recall,
        "Micro F1": micro_f1,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1
    }

def transform_data(x_data, selected_words):        
    transformed = np.zeros((len(x_data), len(selected_words)))
    word_index = {word: i for i, word in enumerate(selected_words)}  # Γρήγορη αναζήτηση των λέξεων
    for i, review in enumerate(x_data):
        indices = [word_index[word] for word in review if word in word_index]  
        transformed[i, indices] = 1
    return transformed
        
def learning_curve(model, X_train_bin, y_train, X_dev_bin, y_dev):
    num_points = 10  # Αριθμός σημείων στον άξονα x
    train_sizes = np.linspace(100, len(X_train_bin), num=num_points, dtype=int)  
    train_precision, train_recall, train_f1 = [], [], []
    dev_precision, dev_recall, dev_f1 = [], [], []

    for size in train_sizes:
        X_train_subset = X_train_bin[:size]
        y_train_subset = y_train[:size]

        model.fit(X_train_subset, y_train_subset)

        y_pred_train = model.predict(X_train_subset)
        y_pred_dev = model.predict(X_dev_bin)

        precision_train, recall_train, f1_train = compute_metrics(y_train_subset, y_pred_train)
        train_precision.append(precision_train)
        train_recall.append(recall_train)
        train_f1.append(f1_train)

        precision_dev, recall_dev, f1_dev = compute_metrics(y_dev, y_pred_dev)
        dev_precision.append(precision_dev)
        dev_recall.append(recall_dev)
        dev_f1.append(f1_dev)

    return train_sizes, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1

def plot_learning_curve(train_sizes, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_sizes, train_precision, label="Train Precision", marker="o")
    plt.plot(train_sizes, dev_precision, label="Dev Precision", marker="s")
    plt.xlabel("Training Size")
    plt.ylabel("Precision")
    plt.title("Learning Curve - Precision")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_sizes, train_recall, label="Train Recall", marker="o")
    plt.plot(train_sizes, dev_recall, label="Dev Recall", marker="s")
    plt.xlabel("Training Size")
    plt.ylabel("Recall")
    plt.title("Learning Curve - Recall")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(train_sizes, train_f1, label="Train F1", marker="o")
    plt.plot(train_sizes, dev_f1, label="Dev F1", marker="s")
    plt.xlabel("Training Size")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve - F1 Score")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main2():
    vocabulary_size = 4000  
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocabulary_size)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    most_frequent_count = int(input("Πόσες από τις πιο συχνές λέξεις θέλεις να εξαιρέσεις; "))
    least_frequent_count = int(input("Πόσες από τις πιο σπάνιες λέξεις θέλεις να εξαιρέσεις; "))
    top_m_words = int(input("Πόσες λέξεις με το υψηλότερο πληροφοριακό κέρδος θέλεις να επιλέξεις; "))

    word_counts = np.zeros(vocabulary_size)
    for review in x_train:
        for word in set(review):  
            if word < vocabulary_size:
                word_counts[word] += 1

    most_frequent_words = np.argsort(word_counts)[-most_frequent_count:]
    least_frequent_words = np.argsort(word_counts)[:least_frequent_count]
    remaining_words = list(set(range(vocabulary_size)) - set(most_frequent_words) - set(least_frequent_words))

    X_train_bin = transform_data(x_train, remaining_words)
    mutual_info = mutual_info_classif(X_train_bin, y_train)

    top_m_words_idx = np.argsort(mutual_info)[-top_m_words:]
    selected_words = [remaining_words[i] for i in top_m_words_idx]

    X_train_bin = transform_data(x_train, selected_words)
    X_dev_bin = transform_data(x_dev, selected_words)
    X_test_bin = transform_data(x_test, selected_words)
    
    bnb = BernoulliNB()
    bnb.fit(X_train_bin, y_train)

    rf = RandomForestClassifier(n_estimators=100, max_depth=20)
    rf.fit(X_train_bin, y_train)

    train_sizes, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1 = learning_curve(bnb, X_train_bin, y_train, X_dev_bin, y_dev)

    plot_learning_curve(train_sizes, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1)
        
    y_pred_bnb = bnb.predict(X_test_bin)
    y_pred_rf = rf.predict(X_test_bin)
    
    bnb_metrics = compute_metricsFullDetails(y_test, y_pred_bnb)
    print("Bernoulli Naive Bayes Metrics:")
    for metric, value in bnb_metrics.items():
        print(f"{metric}: {value}")

    # Υπολογισμός μετρικών για το Random Forest
    rf_metrics = compute_metricsFullDetails(y_test, y_pred_rf)
    print("\nRandom Forest Metrics:")
    for metric, value in rf_metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main2()
