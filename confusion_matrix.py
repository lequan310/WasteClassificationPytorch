import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define the filename
filename = "prediction.csv"

try:
    # 1. Read the data from the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # 2. Extract the true labels (target) and predicted labels
    y_true = df["target"]
    y_pred = df["prediction"]

    # 3. Get the unique class labels to use for the plot axes
    # Sorting ensures the labels on the matrix are in numerical order
    class_labels = sorted(y_true.unique())

    # 4. Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # 5. Create a plot to visualize the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,  # Show the numbers in each cell
        fmt="d",  # Format as integers
        cmap="Blues",  # Color scheme
        xticklabels=class_labels,
        yticklabels=class_labels,
    )

    # Add labels and a title for clarity
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)

    # 6. Display the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    print("Please make sure the script is in the same directory as the CSV file.")
except Exception as e:
    print(f"An error occurred: {e}")
