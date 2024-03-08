from pathlib import Path

from matplotlib import pyplot

from text_loader.loader import DataLoader
from text_loader.visualization import scatter_plot

if __name__ == "__main__":

    # Move to a notebook

    data_loader = DataLoader()
    # Data pre-processing step https://medium.com/analytics-vidhya/predicting-political-orientation-with-machine-learning-be65c950d366
    df = data_loader

    processed_features, labels = data_loader()

    vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8)
    vectors = vectorizer.fit_transform(processed_features).toarray()

    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3)

    text_classifier = RandomForestClassifier()
    text_classifier.fit(X_train, y_train)
    predictions = text_classifier.predict(X_test)

    # Validate model: accuracy and confusion matrix

    # todo: register model step into model folder