from src.data_loader import generate_bangalore_demand
from src.clustering import perform_clustering, find_optimal_clusters
from src.visualize import create_map
from src.utils.helpers import print_project_summary

def main():

    print("Generating commuter demand dataset...")
    data = generate_bangalore_demand()

    print("Determining optimal cluster count using Elbow Method...")
    find_optimal_clusters(data)

    print("Performing K-Means clustering...")
    clustered_data, centroids = perform_clustering(data, n_clusters=8)

    print_project_summary(clustered_data)

    print("Creating interactive optimization map...")
    create_map(clustered_data, centroids)

    print("\nExecution Complete. Check 'outputs' folder.")

if __name__ == "__main__":
    main()