import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    # Ensure all numeric data are in the correct format and handle missing values
    numeric_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)
    return data


def scale_features(data, features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data[features])
    return features_scaled


def find_nearest_neighbors(features_scaled, query_index, num_neighbors):
    nn = NearestNeighbors(n_neighbors=num_neighbors)
    nn.fit(features_scaled)
    distances, indices = nn.kneighbors([features_scaled[query_index]])
    return distances, indices


def plot_neighbors(data, indices, query_index):
    neighbors = data.iloc[indices[0]]
    query_point = data.iloc[query_index]
    fig = px.scatter_3d(neighbors, x='sqft_living', y='bathrooms', z='price', text='price')
    fig.add_scatter3d(x=[query_point['sqft_living']], y=[query_point['bathrooms']], z=[query_point['price']],
                      mode='markers', marker=dict(color='red', size=10), name='Query Point')
    fig.show()


# Main execution flow
if __name__ == "__main__":
    filepath = 'data.csv'
    data = load_and_preprocess_data(filepath)
    features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
    features_scaled = scale_features(data, features)

    # Example: Find neighbors for the first property in the dataset
    query_index = 0  # You can change this index to query different properties
    num_neighbors = 5  # Number of neighbors to find
    distances, indices = find_nearest_neighbors(features_scaled, query_index, num_neighbors)

    print("Distances to Nearest Neighbors:", distances)
    print("Indices of Nearest Neighbors:", indices)

    plot_neighbors(data, indices, query_index)
