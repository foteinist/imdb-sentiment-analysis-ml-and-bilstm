import random
from collections import Counter
from statistics import mode
import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

#Bernoulli Naive Bayes
class BernoulliNaiveBayes:
    def __init__(self):
        self.class_log_prior_ = None  # Λογαριθμικές πιθανότητες εμφάνισης κάθε κλάσης
        self.feature_log_prob_ = None  # Λογαριθμικές πιθανότητες εμφάνισης χαρακτηριστικών ανά κλάση
        self.classes_ = None  # Διαφορετικές κατηγορίες

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)  # Εύρεση των μοναδικών κατηγοριών
        n_classes = len(self.classes_)  # Αριθμός κατηγοριών

        self.class_log_prior_ = np.zeros(n_classes)  # Αρχικοποίηση των λογαριθμικών πιθανοτήτων για κάθε κλάση
        self.feature_log_prob_ = np.zeros((n_classes, n_features))  # Αρχικοποίηση των πιθανοτήτων για τα χαρακτηριστικά

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # Επιλογή δείγματων για κάθε κλάση
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)  # Υπολογισμός λογαριθμικής πιθανότητας για την κλάση
            self.feature_log_prob_[idx] = np.log((X_c.sum(axis=0) + 1) / (X_c.shape[0] + 2))  # Υπολογισμός λογαριθμικών πιθανοτήτων για χαρακτηριστικά
    
    def predict(self, X):
        jll = self._joint_log_likelihood(X)  # Υπολογισμός της κοινής λογαριθμικής πιθανότητας
        return self.classes_[np.argmax(jll, axis=1)]  # Επιστροφή της κλάσης με την μεγαλύτερη πιθανότητα
    
    def _joint_log_likelihood(self, X):
        # 1. Υπολογισμός της πιθανότητας εμφάνισης χαρακτηριστικών (X_i = 1) logP(Xi=1|C)
        log_prob_present = np.dot(X, self.feature_log_prob_.T)
        # 2. Υπολογισμός της πιθανότητας μη εμφάνισης χαρακτηριστικών (X_i = 0)   logP(Xi=0|C)=logP(1-logP(Xi=1|C))
        log_prob_absent = np.dot(1 - X, np.log(1 - np.exp(self.feature_log_prob_)).T)
        # 3. Προσθήκη της λογαριθμικής πιθανότητας της κάθε κατηγορίας (prior) logP(C)
        log_prior = self.class_log_prior_
        # 4. Συνδυασμός όλων των παραπάνω logP(X|C)= logP(C)+S(logP(Xi=1|C)+logP(Xi=0|C))
        joint_log_likelihood = log_prob_present + log_prob_absent + log_prior
        return joint_log_likelihood

    #Κανονικοποίηση
    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)  # Υπολογισμός κοινής λογαριθμικής πιθανότητας
        probs = np.exp(jll)  # Μετατροπή σε πιθανότητες
        probs /= probs.sum(axis=1, keepdims=True)  # Κανονικοποίηση ώστε το άθροισμα των πιθανοτήτων να είναι 1
        return probs

#Node
class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category

#ID3    
class ID3:
    def __init__(self, features, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.tree = None
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    def fit(self, x, y):
        #Create the tree
        if len(y) == 0:
            return None
        most_common = mode(y.flatten())
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common, depth=0)
        return self.tree
    
    def create_tree(self, x_train, y_train, features, category,depth):
        if len(y_train) == 0:
            return Node(is_leaf=True, category=category)
        
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(is_leaf=True, category=mode(y_train.flatten()))

        # check empty data
        if len(x_train) == 0:
            return Node(checking_feature=None, is_leaf=True, category=category)  # decision node
        
        # check all examples belonging in one category
        if np.all(y_train.flatten() == 0):
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.all(y_train.flatten() == 1):
            return Node(checking_feature=None, is_leaf=True, category=1)
        
        if len(features) == 0:
            return Node(checking_feature=None, is_leaf=True, category=mode(y_train.flatten()))
        
        if len(y_train) < self.min_samples_split:
            return Node(is_leaf=True, category=mode(y_train.flatten()))
        
        num_features = len(features)
        selected_features = random.choices(list(features), k=num_features)

        
        igs = [self.calculate_ig(y_train.flatten(), [example[f] for example in x_train]) for f in selected_features]
        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = mode(y_train.flatten())  # most common category 

        root = Node(checking_feature=max_ig_idx)

        # data subset with X = 0
        x_train_0 = x_train[x_train[:, max_ig_idx] == 0, :]
        y_train_0 = y_train[x_train[:, max_ig_idx] == 0].flatten()

        # data subset with X = 1
        x_train_1 = x_train[x_train[:, max_ig_idx] == 1, :]
        y_train_1 = y_train[x_train[:, max_ig_idx] == 1].flatten()

        new_features_indices = np.delete(features.flatten(), max_ig_idx)  # remove current feature

        if len(y_train_0) >= self.min_samples_leaf:
            root.right_child = self.create_tree(x_train=x_train_0, y_train=y_train_0, features=new_features_indices, category=m, depth=depth + 1)
        else:
            root.right_child = Node(is_leaf=True, category=m)
        
        if len(y_train_1) >= self.min_samples_leaf:
            root.left_child = self.create_tree(x_train=x_train_1, y_train=y_train_1, features=new_features_indices, category=m, depth=depth + 1)
        else:
            root.left_child = Node(is_leaf=True, category=m)
        
        return root

    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)

        HC = 0
        for c in classes:
            PC = list(classes_vector).count(c) / len(classes_vector)  # P(C=c)
            HC += - PC * math.log(PC, 2)  # H(C)
            # print('Overall Entropy:', HC)  # entropy for C variable
            
        feature_values = set(feature)  # 0 or 1 in this example
        HC_feature = 0
        for value in feature_values:
            # pf --> P(X=x)
            pf = list(feature).count(value) / len(feature)  # count occurences of value 
            indices = [i for i in range(len(feature)) if feature[i] == value]  # rows (examples) that have X=x

            classes_of_feat = [classes_vector[i] for i in indices]  # category of examples listed in indices above
            for c in classes:
                # pcf --> P(C=c|X=x)
                pcf = classes_of_feat.count(c) / len(classes_of_feat)  # given X=x, count C
                if pcf != 0: 
                    # - P(X=x) * P(C=c|X=x) * log2(P(C=c|X=x))
                    temp_H = - pf * pcf * math.log(pcf, 2)
                    # sum for all values of C (class) and X (values of specific feature)
                    HC_feature += temp_H
        
        ig = HC - HC_feature
        return ig    
     
    def predict(self, x):
        predicted_classes = list()

        for unlabeled in x:  # for every example 
            tmp = self.tree  # begin at root
            while not tmp.is_leaf:
                if unlabeled.flatten()[tmp.checking_feature] == 1:
                    tmp = tmp.left_child
                else:
                    tmp = tmp.right_child
            
            predicted_classes.append(tmp.category)
        
        return np.array(predicted_classes)

#RandomForest
class RandomForest:
    def __init__(self,numberOfTrees=10, max_depth=10,min_samples_split=2, min_samples_leaf=1,feature=None):
        self.trees = []
        self.max_depth = max_depth
        self.numberOfTrees = numberOfTrees
        self.feature = feature
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    def fit(self, x, y):
        '''
        creates the tree
        '''
        self.trees = []
        for _ in range(self.numberOfTrees):
            sample_random = np.random.choice(len(x),len(x), replace=True)
            x_sample = x[sample_random]
            y_sample = y[sample_random]
            
            tree = ID3(features=np.arange(x.shape[1]), max_depth=self.max_depth,
                       min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        final_predictions = []
        for pred in tree_preds:
            counter = Counter(pred)
            most_common = counter.most_common(1)[0][0]
            final_predictions.append(most_common)
        return np.array(final_predictions)

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

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    return precision, recall, f1

def transform_data(x_data, selected_words):        
    transformed = np.zeros((len(x_data), len(selected_words)))
    for i, review in enumerate(x_data):
        transformed[i, [selected_words.index(word) for word in review if word in selected_words]] = 1
    return transformed
           
def learning_curve(bnb, X_train_bin, y_train, X_dev_bin, y_dev):
    num_points = 10  # Αριθμός σημείων στον άξονα x
    train_sizes = np.linspace(100, len(X_train_bin), num=num_points, dtype=int)  # Διάφορα μεγέθη για το training set
    train_precision, train_recall, train_f1 = [], [], []
    dev_precision, dev_recall, dev_f1 = [], [], []

    for size in train_sizes:
        # Χρησιμοποιούμε μόνο το πρώτο 'size' δείγματα για εκπαίδευση
        X_train_subset = X_train_bin[:size]
        y_train_subset = y_train[:size]

        # Εκπαίδευση του Bernoulli Naive Bayes
        bnb.fit(X_train_subset, y_train_subset)

        # Προβλέψεις για τα δεδομένα εκπαίδευσης και ανάπτυξης
        y_pred_train = bnb.predict(X_train_subset)
        y_pred_dev = bnb.predict(X_dev_bin)

        # Υπολογισμός των μετρικών για τα δεδομένα εκπαίδευσης
        precision_train, recall_train, f1_train = compute_metrics(y_train_subset, y_pred_train)
        train_precision.append(precision_train)
        train_recall.append(recall_train)
        train_f1.append(f1_train)

        # Υπολογισμός των μετρικών για τα δεδομένα ανάπτυξης
        precision_dev, recall_dev, f1_dev = compute_metrics(y_dev, y_pred_dev)
        dev_precision.append(precision_dev)
        dev_recall.append(recall_dev)
        dev_f1.append(f1_dev)

    # Επιστροφή των αποτελεσμάτων για την απεικόνιση των καμπυλών
    return train_sizes, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1

def plot_learning_curve(train_sizes, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1):
    plt.figure(figsize=(12, 6))
    
    # Precision Plot
    plt.subplot(1, 3, 1)
    plt.plot(train_sizes, train_precision, label="Train Precision", marker="o")
    plt.plot(train_sizes, dev_precision, label="Dev Precision", marker="s")
    plt.xlabel("Training Size")
    plt.ylabel("Precision")
    plt.title("Learning Curve - Precision")
    plt.legend()
    
    # Recall Plot
    plt.subplot(1, 3, 2)
    plt.plot(train_sizes, train_recall, label="Train Recall", marker="o")
    plt.plot(train_sizes, dev_recall, label="Dev Recall", marker="s")
    plt.xlabel("Training Size")
    plt.ylabel("Recall")
    plt.title("Learning Curve - Recall")
    plt.legend()
    
    # F1 Score Plot
    plt.subplot(1, 3, 3)
    plt.plot(train_sizes, train_f1, label="Train F1", marker="o")
    plt.plot(train_sizes, dev_f1, label="Dev F1", marker="s")
    plt.xlabel("Training Size")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve - F1 Score")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():

    vocabulary_size = 1000  # Μέγεθος λεξιλογίου
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocabulary_size)

    # Διαχωρισμός των δεδομένων σε σύνολα εκπαίδευσης και ανάπτυξης
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Ζητάμε από το χρήστη πόσες από τις πιο συχνές και πιο σπάνιες λέξεις να εξαιρεθούν
    most_frequent_count = int(input("Πόσες από τις πιο συχνές λέξεις θέλεις να εξαιρέσεις; "))
    least_frequent_count = int(input("Πόσες από τις πιο σπάνιες λέξεις θέλεις να εξαιρέσεις; "))
    top_m_words = int(input("Πόσες λέξεις με το υψηλότερο πληροφοριακό κέρδος θέλεις να επιλέξεις; "))

    # Μετράμε τις συχνότητες των λέξεων στο σύνολο εκπαίδευσης
    word_counts = np.zeros(vocabulary_size)
    for review in x_train:
        for word in set(review):  # Χρησιμοποιούμε το set για να μετρήσουμε κάθε λέξη μία φορά ανά έγγραφο
            if word < vocabulary_size:
                word_counts[word] += 1

    # Εντοπίζουμε τις πιο συχνές και τις πιο σπάνιες λέξεις
    most_frequent_words = np.argsort(word_counts)[-most_frequent_count:]
    least_frequent_words = np.argsort(word_counts)[:least_frequent_count]

    # Δημιουργούμε το λεξιλόγιο εξαιρώντας τις συχνές και σπάνιες λέξεις
    remaining_words = list(set(range(vocabulary_size)) - set(most_frequent_words) - set(least_frequent_words))

    # Δημιουργία χαρακτηριστικών για το υπολογισμό του πληροφοριακού κέρδους
    X_train_bin = transform_data(x_train, remaining_words)
   
    # Υπολογισμός του πληροφοριακού κέρδους
    mutual_info = mutual_info_classif(X_train_bin, y_train)

    # Επιλογή των m λέξεων με το υψηλότερο πληροφοριακό κέρδος
    top_m_words_idx = np.argsort(mutual_info)[-top_m_words:]
    selected_words = [remaining_words[i] for i in top_m_words_idx]

    # Μετατρέπουμε τα σύνολα εκπαίδευσης, ανάπτυξης και δοκιμής
    X_train_bin = transform_data(x_train, selected_words)
    X_dev_bin = transform_data(x_dev, selected_words)
    X_test_bin = transform_data(x_test, selected_words)

    # Εκπαίδευση του Bernoulli Naive Bayes
    #print("BernoulliNaiveBayes")
    bnb = BernoulliNaiveBayes()
    bnb.fit(X_train_bin, y_train)

    #print("RandomForest")
    # Εκπαίδευση του Random Forest
    #rf = RandomForest(numberOfTrees=100, max_depth=20)
    #example number 3 numberOfTrees=25, max_depth=15,20,30,100
    rf = RandomForest(numberOfTrees=25, max_depth=20)
    rf.fit(X_train_bin, y_train)

    #Υπολογισμός καμπυλών μάθησης Bernoulli Naive Bayes
    train_sizes, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1 = learning_curve(bnb, X_train_bin, y_train, X_dev_bin, y_dev)

    # Εμφάνιση των καμπυλών μάθησης Bernoulli Naive Bayes
    plot_learning_curve(train_sizes, train_precision, train_recall, train_f1, dev_precision, dev_recall, dev_f1)
    
    y_pred_bnb = bnb.predict(X_test_bin)
    y_pred_rf = rf.predict(X_test_bin)
    
    # Υπολογισμός μετρικών για το Bernoulli Naive Bayes
    print("Bernoulli Naive Bayes Metrics:")
    bnb_metrics = compute_metricsFullDetails(y_test, y_pred_bnb)
    for metric, value in bnb_metrics.items():
        print(f"{metric}: {value}")

    # Υπολογισμός μετρικών για το Random Forest
    print("\nRandom Forest Metrics:")
    rf_metrics = compute_metricsFullDetails(y_test, y_pred_rf)
    for metric, value in rf_metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()