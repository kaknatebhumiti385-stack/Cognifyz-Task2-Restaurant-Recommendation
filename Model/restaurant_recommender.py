
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset
df = pd.read_csv("Dataset/zomato.csv")

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# 2. Clean dataset
df = df.drop_duplicates().copy()

df["City"] = df["City"].astype(str).str.strip()
df["Cuisines"] = df["Cuisines"].astype(str).str.strip()

df["Average Cost for two"] = pd.to_numeric(
    df["Average Cost for two"], errors="coerce"
)
df["Price range"] = pd.to_numeric(
    df["Price range"], errors="coerce"
)
df["Aggregate rating"] = pd.to_numeric(
    df["Aggregate rating"], errors="coerce"
)

df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0)

df = df.dropna(
    subset=["City", "Cuisines", "Average Cost for two",
            "Price range", "Aggregate rating"]
).copy()

# 3. Create recommendation features
df["Features"] = (
    df["City"] + " " +
    df["Cuisines"] + " " +
    df["Price range"].astype(str)
)

# 4. Convert text into TF-IDF features
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df["Features"])

# 5. Calculate cosine similarity
similarity_matrix = cosine_similarity(feature_matrix)

print("Recommendation model created successfully!")

# 6. Restaurant recommendation function
def recommend_restaurants(city=None, cuisine=None,
                          price_range=None, top_n=5):

    preferences = []

    if city:
        preferences.append(str(city).strip())

    if cuisine:
        preferences.append(str(cuisine).strip())

    if price_range is not None:
        preferences.append(str(price_range))

    if not preferences:
        print("Please enter at least one preference.")
        return pd.DataFrame()

    query = " ".join(preferences)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector, feature_matrix
    ).flatten()

    results = df.copy()
    results["Similarity Score"] = scores

    # Apply exact preference filters
    if city:
        results = results[
            results["City"].str.lower() == str(city).strip().lower()
        ]

    if cuisine:
        results = results[
            results["Cuisines"].str.lower() == str(cuisine).strip().lower()
        ]

    if price_range is not None:
        results = results[
            results["Price range"] == price_range
        ]

    if results.empty:
        print("No restaurants found for these preferences.")
        return pd.DataFrame()

    # Calculate rating score
    results["Rating Score"] = (
        results["Aggregate rating"].clip(0, 5) / 5
    )

    # Calculate normalized vote score
    results["Vote Score"] = (
        np.log1p(results["Votes"]) /
        np.log1p(max(df["Votes"].max(), 1))
    )

    # Calculate combined recommendation score
    results["Recommendation Score"] = (
        0.45 * results["Similarity Score"] +
        0.40 * results["Rating Score"] +
        0.15 * results["Vote Score"]
    )

    results = results.sort_values(
        by="Recommendation Score",
        ascending=False
    )

    columns = [
        "Restaurant ID", "City", "Cuisines",
        "Average Cost for two", "Price range",
        "Aggregate rating", "Votes",
        "Similarity Score", "Recommendation Score"
    ]

    return results[columns].head(top_n)

   


# 7. Save model
os.makedirs("Model", exist_ok=True)
joblib.dump(vectorizer, "Model/tfidf_vectorizer.pkl")
joblib.dump(similarity_matrix, "Model/similarity_matrix.pkl")

# 8. Save cleaned dataset
os.makedirs("Output", exist_ok=True)
df.to_csv("Output/cleaned_zomato.csv", index=False)

# 9. Generate reports
os.makedirs("Reports", exist_ok=True)

# Top cuisines chart
plt.figure(figsize=(10, 5))
df["Cuisines"].value_counts().plot(kind="bar")
plt.title("Restaurant Distribution by Cuisine")
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Reports/cuisine_distribution.png")
plt.close()

# Rating distribution chart
plt.figure(figsize=(8, 5))
df["Aggregate rating"].plot(kind="hist", bins=10)
plt.title("Restaurant Rating Distribution")
plt.xlabel("Aggregate Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("Reports/rating_distribution.png")
plt.close()

# City distribution chart
plt.figure(figsize=(10, 5))
df["City"].value_counts().plot(kind="bar")
plt.title("Restaurant Distribution by City")
plt.xlabel("City")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Reports/city_distribution.png")
plt.close()

# 10. Test recommendation system
recommendations = recommend_restaurants(
    city="New Delhi",
    cuisine="Cafe",
    price_range=2,
    top_n=5
)

print("\nTEST RECOMMENDATIONS:")
print(recommendations.to_string(index=False))

recommendations.to_csv(
    "Output/test_recommendations.csv", index=False
)

print("\nAll operations completed successfully!")
print("Models saved in Model folder.")
print("Charts saved in Reports folder.")
print("Results saved in Output folder.")
