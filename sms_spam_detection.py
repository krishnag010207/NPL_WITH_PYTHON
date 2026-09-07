import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Task 1 – Load and Explore
df = pd.read_excel("spam .xlsx")
df = df[["Category", "Message"]].copy()
df["Category"] = df["Category"].astype(str).str.strip().str.lower()
df["Message"] = df["Message"].fillna("").astype(str)

print(df.head())
print("Rows and columns:", df.shape)
print(df["Category"].value_counts())

# Task 2 – Visualize
counts = df["Category"].value_counts()
plt.bar(counts.index, counts.values)
plt.title("Number of Spam vs Ham Messages")
plt.xlabel("Message Category")
plt.ylabel("Number of Messages")
plt.show()

# Task 3 – Prepare Data
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})
X = df["Message"]
y = df["Category"]

# Task 4 – Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Task 5 – Convert Text into Numbers
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Task 6 – Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Task 7 – Make Predictions
predictions = model.predict(X_test_vec)

# Task 8 – Evaluate
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
).plot()
plt.title("SMS Spam Detection – Confusion Matrix")
plt.show()

# Task 9 – Test Your Own SMS
new_message = input("Enter an SMS message: ")
new_message_vector = vectorizer.transform([new_message])
prediction = model.predict(new_message_vector)[0]

if prediction == 0:
    print("📩 HAM MESSAGE")
else:
    print("🚨 SPAM MESSAGE")
