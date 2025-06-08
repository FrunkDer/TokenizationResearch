import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from gensim.models import Word2Vec
import time

# Paths
lang = "ZFinnish"  # Choose language
#WHEN DOING FINNISH ADD WORD TO THE LIST
strategies = ["SUBWORDCORRECTEDBPE10k", "SUBWORDCORRECTEDBPE25k", "SUBWORDCORRECTEDBPE50k", "SUBWORDCORRECTEDBPE5k"]  #PUT WORD BACK IN 
for strategy in strategies:
    model_path = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Word2Vec"
    if lang == "ZFinnish" or lang == "Finnish":
        train_file = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\Evaluation\Finnishtrain.conll"
        test_file = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\Evaluation\Finnishtest.conll"
    else:
        train_file = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\Evaluation\Turkishtrain.conll"
        test_file = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\Evaluation\Turkishtest.conll"
    output_stats_path = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Evaluation\{lang}_{strategy}_POS_results.json"

    # Load Word2Vec model
    def load_word2vec():
        model_file = os.path.join(model_path, f"{lang}_{strategy}_word2vec.model")
        print(f"Loading Word2Vec model: {model_file}")
        return Word2Vec.load(model_file)

    # Parse CoNLL files
    def load_conll_data(file_path):
        sentences = []
        pos_tags = []
        sentence = []
        tags = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:  # Token and tag are present
                    parts = line.split("\t")
                    if len(parts) == 2:
                        word, tag = parts
                        sentence.append(word)
                        tags.append(tag)
                else:  # Sentence boundary
                    if sentence:
                        sentences.append(sentence)
                        pos_tags.append(tags)
                        sentence = []
                        tags = []

        return sentences, pos_tags

    # Convert words to embeddings
    def words_to_embeddings(sentences, pos_tags, w2v_model):
        X = []
        y = []

        for sentence, tags in zip(sentences, pos_tags):
            for word, tag in zip(sentence, tags):
                if word in w2v_model.wv:
                    X.append(w2v_model.wv[word])  # Word vector
                else:
                    X.append(np.zeros(w2v_model.vector_size))  # OOV handling
                y.append(tag)

        return np.array(X), np.array(y)

    # Train and evaluate Logistic Regression model
    def train_logistic_regression(X_train, y_train, X_test, y_test):
        print("Encoding POS tags...")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        print("Training Logistic Regression...")
        start_time = time.time()
        model = LogisticRegression(max_iter=500, solver="saga", verbose=1, n_jobs=-1)
        model.fit(X_train, y_train_encoded)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")

        print("Evaluating model...")
        y_pred = model.predict(X_test)

        # Compute performance metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        class_report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True)
        conf_matrix = confusion_matrix(y_test_encoded, y_pred)

        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
        
        # Save results to file
        save_results(accuracy, class_report, conf_matrix, label_encoder.classes_)

    # Save evaluation results to a file
    def save_results(accuracy, class_report, conf_matrix, class_labels):
        results = {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "labels": class_labels.tolist()
        }

        with open(output_stats_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {output_stats_path}")

    # Load Word2Vec model
    w2v_model = load_word2vec()

    # Load and process dataset
    print("Loading training data...")
    train_sentences, train_pos_tags = load_conll_data(train_file)
    X_train, y_train = words_to_embeddings(train_sentences, train_pos_tags, w2v_model)

    print("Loading testing data...")
    test_sentences, test_pos_tags = load_conll_data(test_file)
    X_test, y_test = words_to_embeddings(test_sentences, test_pos_tags, w2v_model)

    # Train and evaluate
    train_logistic_regression(X_train, y_train, X_test, y_test)
