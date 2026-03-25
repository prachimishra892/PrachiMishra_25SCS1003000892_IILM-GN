import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("music_dataset.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

# ------------------------------
# 2. Select Features
# ------------------------------
features = [
    "danceability", "energy", "valence", "tempo",
    "loudness", "acousticness", "instrumentalness"
]

X = df[features]
y = df["genre"]

# ------------------------------
# 3. Preprocessing
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 4. Train Model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# 5. Accuracy
# ------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Trained Successfully!")
print("Accuracy:", accuracy)

# ------------------------------
# 6. Recommendation Function
# ------------------------------
def recommend(song_index):
    similarity = cosine_similarity([X_scaled[song_index]], X_scaled)[0]
    similar_indices = similarity.argsort()[::-1][1:6]

    print("\nRecommended Songs:\n")
    for i in similar_indices:
        print(df.iloc[i]["song_name"], "-", df.iloc[i]["artist"])

# ------------------------------
# 7. Example Recommendation
# ------------------------------
print("\nRecommendations for:", df.iloc[0]["song_name"])
recommend(0)