from tkinter import messagebox
from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# CNN is implemented with PyTorch so this version can run on Python 3.14.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------
# Global variables
# ---------------------------------------------------------
main = None

filename = ""
dataset = None

X = None
Y = None
X_train = None
X_test = None
y_train = None
y_test = None

feature_encoders = {}
target_encoder = None
feature_columns = []

classifier = None
hist = None

accuracy = []
precision = []
recall = []
fscore = []
sensitivity = []
specificity = []


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def show_error(title, message):
    messagebox.showerror(title, message)


def check_data_processed():
    if X_train is None or X_test is None:
        show_error(
            "Data not processed",
            "Please upload the ASD dataset and click 'Preprocess Data' first."
        )
        return False
    return True


def safe_transform(encoder, values):
    """
    Transform categorical values using a previously fitted LabelEncoder.
    Unknown categories are assigned to 0 instead of crashing.
    """
    mapping = {str(value): index for index, value in enumerate(encoder.classes_)}
    return np.array(
        [mapping.get(str(value), 0) for value in values],
        dtype=np.float32
    )


def calculateMetrics(algorithm, predict, testY):
    global accuracy, precision, recall, fscore, sensitivity, specificity

    predict = np.asarray(predict)
    testY = np.asarray(testY)

    p = precision_score(
        testY, predict, average="macro", zero_division=0
    ) * 100
    r = recall_score(
        testY, predict, average="macro", zero_division=0
    ) * 100
    f = f1_score(
        testY, predict, average="macro", zero_division=0
    ) * 100
    a = accuracy_score(testY, predict) * 100

    cm = confusion_matrix(testY, predict, labels=[0, 1])

    # For this project:
    # class 0 = No Autism
    # class 1 = Autism
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity = recall for the positive class (autism)
    se = (tp / (tp + fn) * 100) if (tp + fn) else 0.0
    # Specificity = true negative rate for the negative class (no autism)
    sp = (tn / (tn + fp) * 100) if (tn + fp) else 0.0

    text.insert(END, algorithm + " Accuracy       : " + f"{a:.2f}" + "\n")
    text.insert(END, algorithm + " Precision      : " + f"{p:.2f}" + "\n")
    text.insert(END, algorithm + " Recall         : " + f"{r:.2f}" + "\n")
    text.insert(END, algorithm + " FScore         : " + f"{f:.2f}" + "\n")
    text.insert(END, algorithm + " Sensitivity    : " + f"{se:.2f}" + "\n")
    text.insert(END, algorithm + " Specificity    : " + f"{sp:.2f}" + "\n\n")

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    sensitivity.append(se)
    specificity.append(sp)

    text.update_idletasks()

    LABELS = ["No", "Yes"]
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        cm,
        xticklabels=LABELS,
        yticklabels=LABELS,
        annot=True,
        cmap="viridis",
        fmt="g"
    )
    ax.set_ylim([0, 2])
    plt.title(algorithm + " Confusion Matrix")
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Upload dataset
# ---------------------------------------------------------
def upload():
    global filename, dataset

    filename = filedialog.askopenfilename(
        initialdir="Dataset",
        title="Select ASD Dataset",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not filename:
        return

    try:
        dataset = pd.read_csv(filename)

        if "Class/ASD" not in dataset.columns:
            show_error(
                "Invalid Dataset",
                "The selected CSV must contain the 'Class/ASD' column."
            )
            return

        pathlabel.config(text=filename)
        text.delete("1.0", END)
        text.insert(END, filename + " loaded\n\n")
        text.insert(END, str(dataset.head()) + "\n")

        label = dataset.groupby("Class/ASD").size()

        plt.figure(figsize=(6, 4))
        label.plot(kind="bar")
        plt.title("With & Without Autism Disorder")
        plt.xlabel("Class")
        plt.ylabel("Number of Records")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        show_error("Upload Error", str(e))


# ---------------------------------------------------------
# Preprocess dataset
# ---------------------------------------------------------
def processDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    global feature_encoders, target_encoder, feature_columns
    global accuracy, precision, recall, fscore, sensitivity, specificity
    global classifier, hist

    if dataset is None:
        show_error(
            "Dataset Missing",
            "Please upload Autism-Adult-Data.csv first."
        )
        return

    try:
        text.delete("1.0", END)

        df = dataset.copy()
        df = df.replace(np.nan, 0)
        df = df.fillna(0)

        if "Class/ASD" not in df.columns:
            show_error(
                "Invalid Dataset",
                "The dataset must contain the 'Class/ASD' target column."
            )
            return

        target_column = "Class/ASD"
        feature_columns = [c for c in df.columns if c != target_column]

        # Convert feature columns to numeric.
        # Numeric columns remain numeric; categorical columns are label encoded.
        feature_encoders = {}

        for column in feature_columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(
                    df[column].astype(str)
                )
                feature_encoders[column] = encoder
            else:
                df[column] = pd.to_numeric(
                    df[column], errors="coerce"
                ).fillna(0)

        # Encode target: NO -> 0, YES -> 1
        target_encoder = LabelEncoder()
        df[target_column] = target_encoder.fit_transform(
            df[target_column].astype(str)
        )

        X = df[feature_columns].to_numpy(dtype=np.float32)
        Y = df[target_column].to_numpy(dtype=np.int64)

        # Fixed random_state makes results reproducible.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            Y,
            test_size=0.20,
            random_state=42,
            stratify=Y
        )

        # Reset previous model/results after preprocessing.
        classifier = None
        hist = None

        accuracy = []
        precision = []
        recall = []
        fscore = []
        sensitivity = []
        specificity = []

        text.insert(END, str(df.head()) + "\n\n")
        text.insert(
            END,
            "Total records found in dataset are : "
            + str(X.shape[0]) + "\n"
        )
        text.insert(
            END,
            "Total records used to train machine learning algorithms are : "
            + str(X_train.shape[0]) + "\n"
        )
        text.insert(
            END,
            "Total records used to test machine learning algorithms are : "
            + str(X_test.shape[0]) + "\n\n"
        )

        text.insert(
            END,
            "Preprocessing completed successfully.\n"
        )
        text.insert(
            END,
            "Features used : " + str(len(feature_columns)) + "\n"
        )
        text.insert(
            END,
            "Target classes : " + str(list(target_encoder.classes_)) + "\n"
        )

    except Exception as e:
        show_error("Preprocessing Error", str(e))


# ---------------------------------------------------------
# SVM
# ---------------------------------------------------------
def runSVM():
    global accuracy, precision, recall, fscore, sensitivity, specificity

    if not check_data_processed():
        return

    try:
        text.delete("1.0", END)

        # Keep SVM as the first algorithm in the performance arrays.
        accuracy = []
        precision = []
        recall = []
        fscore = []
        sensitivity = []
        specificity = []

        svm_cls = svm.SVC()
        svm_cls.fit(X_train, y_train)

        train_predict = svm_cls.predict(X_train)
        train_acc = accuracy_score(y_train, train_predict) * 100

        text.insert(
            END,
            "SVM Training Accuracy: "
            + f"{train_acc:.2f}"
            + "\n\n"
        )

        test_predict = svm_cls.predict(X_test)
        calculateMetrics("SVM", test_predict, y_test)

    except Exception as e:
        show_error("SVM Error", str(e))


# ---------------------------------------------------------
# KNN
# ---------------------------------------------------------
def runKNN():
    if not check_data_processed():
        return

    try:
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X_train, y_train)

        train_predict = knn.predict(X_train)
        train_acc = accuracy_score(y_train, train_predict) * 100

        text.insert(
            END,
            "KNN Training Accuracy: "
            + f"{train_acc:.2f}"
            + "\n\n"
        )

        predict = knn.predict(X_test)
        calculateMetrics("KNN", predict, y_test)

    except Exception as e:
        show_error("KNN Error", str(e))


# ---------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------
def runNaiveBayes():
    if not check_data_processed():
        return

    try:
        nb = GaussianNB()
        nb.fit(X_train, y_train)

        train_predict = nb.predict(X_train)
        train_acc = accuracy_score(y_train, train_predict) * 100

        text.insert(
            END,
            "Naive Bayes Training Accuracy: "
            + f"{train_acc:.2f}"
            + "\n\n"
        )

        predict = nb.predict(X_test)
        calculateMetrics("Naive Bayes", predict, y_test)

    except Exception as e:
        show_error("Naive Bayes Error", str(e))


# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------
def runlogisticRegression():
    if not check_data_processed():
        return

    try:
        lr = LogisticRegression(
            solver="liblinear",
            max_iter=1000
        )
        lr.fit(X_train, y_train)

        train_predict = lr.predict(X_train)
        train_acc = accuracy_score(y_train, train_predict) * 100

        text.insert(
            END,
            "Logistic Regression Training Accuracy: "
            + f"{train_acc:.2f}"
            + "\n\n"
        )

        predict = lr.predict(X_test)

        calculateMetrics(
            "Logistic Regression",
            predict,
            y_test
        )

    except Exception as e:
        show_error("Logistic Regression Error", str(e))


# ---------------------------------------------------------
# ANN
# ---------------------------------------------------------
def runANN():
    if not check_data_processed():
        return

    try:
        ann = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=42
        )

        ann.fit(X_train, y_train)

        train_predict = ann.predict(X_train)
        train_acc = accuracy_score(y_train, train_predict) * 100

        text.insert(
            END,
            "ANN Training Accuracy: "
            + f"{train_acc:.2f}"
            + "\n\n"
        )

        predict = ann.predict(X_test)
        calculateMetrics("ANN", predict, y_test)

    except Exception as e:
        show_error("ANN Error", str(e))


# ---------------------------------------------------------
# CNN model
# ---------------------------------------------------------
class ASD_CNN(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------
# Run CNN
# ---------------------------------------------------------
def runCNN():
    global classifier, hist

    if not check_data_processed():
        return

    if not TORCH_AVAILABLE:
        show_error(
            "PyTorch Missing",
            "PyTorch is required for CNN.\n\n"
            "Install it using:\n"
            "pip install torch"
        )
        return

    try:
        text.delete("1.0", END)
        text.insert(END, "CNN Algorithm - PyTorch 1D CNN\n\n")
        text.insert(
            END,
            "Preparing training and testing data...\n"
        )
        text.update_idletasks()

        # Convert tabular data to CNN format:
        # [samples, features] -> [samples, 1, features]
        X_train_tensor = torch.tensor(
            X_train,
            dtype=torch.float32
        ).unsqueeze(1)

        X_test_tensor = torch.tensor(
            X_test,
            dtype=torch.float32
        ).unsqueeze(1)

        y_train_tensor = torch.tensor(
            y_train,
            dtype=torch.long
        )

        y_test_tensor = torch.tensor(
            y_test,
            dtype=torch.long
        )

        train_dataset = TensorDataset(
            X_train_tensor,
            y_train_tensor
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        text.insert(
            END,
            "CNN Device: " + str(device) + "\n"
        )

        text.insert(
            END,
            "Input shape: "
            + str(tuple(X_train_tensor.shape))
            + "\n\n"
        )

        model = ASD_CNN(
            input_features=X_train.shape[1]
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        epochs = 30

        train_accuracy_history = []
        train_loss_history = []
        validation_accuracy_history = []
        validation_loss_history = []

        # Training
        for epoch in range(epochs):
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                running_loss += (
                    loss.item() * batch_X.size(0)
                )

                predicted = torch.argmax(
                    outputs,
                    dim=1
                )

                correct += (
                    predicted == batch_y
                ).sum().item()

                total += batch_y.size(0)

            epoch_loss = (
                running_loss / total
                if total else 0
            )

            epoch_accuracy = (
                correct / total * 100
                if total else 0
            )

            # Validation
            model.eval()

            with torch.no_grad():
                validation_outputs = model(
                    X_test_tensor.to(device)
                )

                validation_loss = criterion(
                    validation_outputs,
                    y_test_tensor.to(device)
                ).item()

                validation_predicted = torch.argmax(
                    validation_outputs,
                    dim=1
                )

                validation_accuracy = (
                    (
                        validation_predicted
                        == y_test_tensor.to(device)
                    ).sum().item()
                    / len(y_test_tensor)
                    * 100
                )

            train_loss_history.append(epoch_loss)
            train_accuracy_history.append(epoch_accuracy)
            validation_loss_history.append(validation_loss)
            validation_accuracy_history.append(
                validation_accuracy
            )

            if (
                (epoch + 1) % 5 == 0
                or epoch == 0
                or epoch == epochs - 1
            ):
                text.insert(
                    END,
                    "Epoch "
                    + str(epoch + 1)
                    + "/"
                    + str(epochs)
                    + " - Loss: "
                    + f"{epoch_loss:.4f}"
                    + " - Accuracy: "
                    + f"{epoch_accuracy:.2f}%"
                    + " - Val Accuracy: "
                    + f"{validation_accuracy:.2f}%"
                    + "\n"
                )
                text.see(END)
                text.update_idletasks()

        classifier = model

        hist = {
            "accuracy": train_accuracy_history,
            "loss": train_loss_history,
            "val_accuracy": validation_accuracy_history,
            "val_loss": validation_loss_history
        }

        # Final CNN prediction
        model.eval()

        with torch.no_grad():
            outputs = model(
                X_test_tensor.to(device)
            )

            predict = torch.argmax(
                outputs,
                dim=1
            ).cpu().numpy()

        train_accuracy = train_accuracy_history[-1]

        text.insert(
            END,
            "\nCNN Training Accuracy: "
            + f"{train_accuracy:.2f}%"
            + "\n\n"
        )

        calculateMetrics(
            "CNN",
            predict,
            y_test
        )

        text.insert(
            END,
            "CNN training completed successfully.\n"
        )

    except Exception as e:
        show_error(
            "CNN Error",
            str(e)
        )


# ---------------------------------------------------------
# Autism detection using trained CNN
# ---------------------------------------------------------
def detectAutism():
    global classifier

    if classifier is None:
        show_error(
            "CNN Model Missing",
            "Please run 'Run CNN Algorithm' first."
        )
        return

    if not TORCH_AVAILABLE:
        show_error(
            "PyTorch Missing",
            "Install PyTorch using: pip install torch"
        )
        return

    try:
        text.delete("1.0", END)

        test_filename = filedialog.askopenfilename(
            initialdir="Dataset",
            title="Select Test Data",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not test_filename:
            return

        testData = pd.read_csv(test_filename)
        testData = testData.replace(np.nan, 0)
        testData = testData.fillna(0)

        missing_columns = [
            c for c in feature_columns
            if c not in testData.columns
        ]

        if missing_columns:
            show_error(
                "Invalid Test Data",
                "The following feature columns are missing:\n"
                + ", ".join(missing_columns)
            )
            return

        # Keep exactly the same feature order as training.
        test_features = testData[feature_columns].copy()

        for column in feature_columns:
            if column in feature_encoders:
                test_features[column] = safe_transform(
                    feature_encoders[column],
                    test_features[column].astype(str)
                )
            else:
                test_features[column] = pd.to_numeric(
                    test_features[column],
                    errors="coerce"
                ).fillna(0)

        test_array = test_features.to_numpy(
            dtype=np.float32
        )

        test_tensor = torch.tensor(
            test_array,
            dtype=torch.float32
        ).unsqueeze(1)

        device = next(
            classifier.parameters()
        ).device

        classifier.eval()

        with torch.no_grad():
            outputs = classifier(
                test_tensor.to(device)
            )

            predict = torch.argmax(
                outputs,
                dim=1
            ).cpu().numpy()

        # Use target encoder to display the actual class names.
        class_names = list(
            target_encoder.classes_
        )

        text.insert(
            END,
            "Autism Detection Results\n\n"
        )

        for i, prediction in enumerate(predict):
            predicted_class = class_names[int(prediction)]

            if predicted_class.upper() == "YES":
                label = "Autism Disorder Detected"
            else:
                label = "No Autism Disorder Detected"

            text.insert(
                END,
                "Test Data "
                + str(i + 1)
                + " =====> Predicted Output : "
                + label
                + " ("
                + predicted_class
                + ")\n\n"
            )

    except Exception as e:
        show_error(
            "Detection Error",
            str(e)
        )


# ---------------------------------------------------------
# All algorithms performance graph
# ---------------------------------------------------------
def graph():
    if len(accuracy) < 6:
        text.delete("1.0", END)
        text.insert(
            END,
            "Please run all 6 algorithms first:\n"
            "SVM, KNN, Naive Bayes, Logistic Regression, ANN and CNN.\n"
        )
        return

    try:
        algorithms = [
            "SVM",
            "KNN",
            "Naive Bayes",
            "Logistic Regression",
            "ANN",
            "CNN"
        ]

        rows = []

        for i, algorithm in enumerate(algorithms):
            rows.extend([
                [algorithm, "Precision", precision[i]],
                [algorithm, "Recall", recall[i]],
                [algorithm, "F1 Score", fscore[i]],
                [algorithm, "Accuracy", accuracy[i]],
                [algorithm, "Sensitivity", sensitivity[i]],
                [algorithm, "Specificity", specificity[i]]
            ])

        df = pd.DataFrame(
            rows,
            columns=[
                "Algorithms",
                "Parameters",
                "Value"
            ]
        )

        df.to_csv(
            "aa.csv",
            index=False
        )

        pivot_df = df.pivot(
            index="Parameters",
            columns="Algorithms",
            values="Value"
        )

        plt.figure(figsize=(13, 7))
        pivot_df.plot(kind="bar", figsize=(13, 7))
        plt.title(
            "All Machine Learning Algorithms Performance"
        )
        plt.ylabel("Percentage")
        plt.xlabel("Performance Parameters")
        plt.xticks(rotation=0)
        plt.legend(
            title="Algorithms",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )
        plt.tight_layout()
        plt.show()

    except Exception as e:
        show_error(
            "Graph Error",
            str(e)
        )


# ---------------------------------------------------------
# CNN training graph
# ---------------------------------------------------------
def cnngraph():
    global hist

    if hist is None:
        show_error(
            "CNN Training Graph",
            "Please run the CNN algorithm first."
        )
        return

    try:
        epochs = range(
            1,
            len(hist["accuracy"]) + 1
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs,
            hist["accuracy"],
            marker="o",
            label="CNN Training Accuracy"
        )
        plt.plot(
            epochs,
            hist["val_accuracy"],
            marker="o",
            label="CNN Validation Accuracy"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(
            "CNN Training and Validation Accuracy"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs,
            hist["loss"],
            marker="o",
            label="CNN Training Loss"
        )
        plt.plot(
            epochs,
            hist["val_loss"],
            marker="o",
            label="CNN Validation Loss"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            "CNN Training and Validation Loss"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        show_error(
            "CNN Graph Error",
            str(e)
        )


# ---------------------------------------------------------
# GUI
# ---------------------------------------------------------
if __name__ == "__main__":
    main = tkinter.Tk()
    main.title(
        "Analysis and Detection of Autism Spectrum Disorder Using Machine Learning Techniques"
    )
    main.geometry("1300x1200")

    font = ("times", 14, "bold")

    title = Label(
        main,
        text=(
            "Analysis and Detection of Autism Spectrum Disorder "
            "Using Machine Learning Techniques"
        )
    )
    title.config(
        bg="yellow3",
        fg="white",
        font=font,
        height=3,
        width=120
    )
    title.place(x=0, y=5)

    font1 = ("times", 13, "bold")

    uploadButton = Button(
        main,
        text="Upload ASD Dataset",
        command=upload
    )
    uploadButton.place(x=50, y=100)
    uploadButton.config(font=font1)

    pathlabel = Label(main)
    pathlabel.config(
        bg="brown",
        fg="white",
        font=font1
    )
    pathlabel.place(x=460, y=100)

    processButton = Button(
        main,
        text="Preprocess Data",
        command=processDataset
    )
    processButton.place(x=50, y=150)
    processButton.config(font=font1)

    svmButton = Button(
        main,
        text="Run SVM Algorithm",
        command=runSVM
    )
    svmButton.place(x=280, y=150)
    svmButton.config(font=font1)

    knnButton = Button(
        main,
        text="Run KNN Algorithm",
        command=runKNN
    )
    knnButton.place(x=530, y=150)
    knnButton.config(font=font1)

    nbbutton = Button(
        main,
        text="Run NaiveBayes Machine",
        command=runNaiveBayes
    )
    nbbutton.place(x=730, y=150)
    nbbutton.config(font=font1)

    lrButton = Button(
        main,
        text="Run Logistic Regression",
        command=runlogisticRegression
    )
    lrButton.place(x=50, y=200)
    lrButton.config(font=font1)

    annButton = Button(
        main,
        text="Run ANN Algorithm",
        command=runANN
    )
    annButton.place(x=280, y=200)
    annButton.config(font=font1)

    cnnButton = Button(
        main,
        text="Run CNN Algorithm",
        command=runCNN
    )
    cnnButton.place(x=530, y=200)
    cnnButton.config(font=font1)

    detectButton = Button(
        main,
        text="Detect Autism from Test Data",
        command=detectAutism
    )
    detectButton.place(x=730, y=200)
    detectButton.config(font=font1)

    graphButton = Button(
        main,
        text="All Algorithms Performance Graph",
        command=graph
    )
    graphButton.place(x=50, y=250)
    graphButton.config(font=font1)

    cnngraphButton = Button(
        main,
        text="CNN Training Graph",
        command=cnngraph
    )
    cnngraphButton.place(x=360, y=250)
    cnngraphButton.config(font=font1)

    font1 = ("times", 12, "bold")

    text = Text(
        main,
        height=20,
        width=150
    )
    text.place(x=10, y=300)
    text.config(font=font1)

    scroll = Scrollbar(main, command=text.yview)
    scroll.place(
        x=1270,
        y=300,
        height=325
    )
    text.configure(
        yscrollcommand=scroll.set
    )

    main.config(bg="burlywood2")
    main.mainloop()