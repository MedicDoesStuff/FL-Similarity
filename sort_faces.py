import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster
from fastcluster import linkage
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import InceptionResnetV1
import os
import shutil
from PIL import Image

# Verify PyTorch version
print("PyTorch Version:", torch.__version__)

# Define paths (modify these as needed for your environment)
Data = "data_src"  # Change to "data_dst" if needed
image_folder = f'workspace/{Data}/aligned'  # Adjust path for your setup
output_folder = 'clustered_faces'

# Create output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get image paths
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_paths:
    raise ValueError(f"No images found in {image_folder}. Please upload images.")

# Initialize InceptionResnetV1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
except Exception as e:
    print(f"Error initializing InceptionResnetV1: {e}")
    raise

# Load and preprocess pre-cropped image
def load_and_preprocess_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB').resize((160, 160))
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC to CHW
        return img_tensor
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Extract embeddings
embeddings = []
valid_image_paths = []
for img_path in image_paths:
    img_tensor = load_and_preprocess_image(img_path)
    if img_tensor is not None:
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(img_tensor)[0].cpu().numpy()
        # Normalize embedding to unit length
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
        valid_image_paths.append(img_path)
embeddings = np.array(embeddings)

if len(embeddings) == 0:
    raise ValueError("No valid images processed. Check image formats and paths.")

# Compute distance matrix using cosine distance
dist_mat = squareform(pdist(embeddings, metric='cosineKILL

# Visualize raw distance matrix
N = len(embeddings)
plt.figure(figsize=(8, 8))
plt.pcolormesh(dist_mat)
plt.colorbar()
plt.xlim([0, N])
plt.ylim([0, N])
plt.title("Raw Cosine Distance Matrix")
plt.savefig(os.path.join(output_folder, 'raw_distance_matrix.png'))
plt.close()

# Clustering functions
def seriation(Z, N, cur_index):
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)

def compute_serial_matrix(dist_mat, method="average"):
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]
    return seriated_dist, res_order, res_linkage

# Sort images and compute clusters
method = "average"
distance_threshold = 0.8  # Lowered for stricter clustering
print(f"Sorting and clustering with Method: {method}, Distance Threshold: {distance_threshold}")
ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)

# Assign images to clusters based on distance threshold
cluster_labels = fcluster(res_linkage, t=distance_threshold, criterion='distance')
n_clusters = len(np.unique(cluster_labels))

print(f"Number of clusters formed: {n_clusters}")
sorted_images = [valid_image_paths[i] for i in res_order]
sorted_labels = [cluster_labels[i] for i in res_order]

# Visualize sorted distance matrix
plt.figure(figsize=(8, 8))
plt.pcolormesh(ordered_dist_mat)
plt.xlim([0, N])
plt.ylim([0, N])
plt.title(f"Sorted Cosine Distance Matrix ({method})")
plt.savefig(os.path.join(output_folder, f'sorted_distance_matrix_{method}.png'))
plt.close()

# Organize images into cluster folders
cluster_folders = {}
new_image_paths = []
for cluster_id in range(1, n_clusters + 1):
    cluster_folder = os.path.join(output_folder, f'cluster_{cluster_id}')
    os.makedirs(cluster_folder, exist_ok=True)
    cluster_folders[cluster_id] = cluster_folder

for img_path, cluster_id in zip(sorted_images, sorted_labels):
    new_filename = os.path.basename(img_path)
    new_path = os.path.join(cluster_folders[cluster_id], new_filename)
    shutil.copy(img_path, new_path)
    new_image_paths.append((new_path, cluster_id))

# Print sorted and clustered image paths
print("Sorted and Clustered Image Paths:")
for path, cluster_id in new_image_paths:
    print(f"Cluster {cluster_id}: {path}")

# Visualize images from each cluster
def display_cluster_images(cluster_folders, image_paths, max_images=5):
    for cluster_id, folder in cluster_folders.items():
        cluster_images = [path for path, cid in image_paths if cid == cluster_id][:max_images]
        if not cluster_images:
            continue
        plt.figure(figsize=(15, 5))
        for i, img_path in enumerate(cluster_images):
            try:
                img = Image.open(img_path).resize((112, 112))
                plt.subplot(1, min(max_images, len(cluster_images)), i + 1)
                plt.imshow(img)
                plt.title(os.path.basename(img_path), fontsize=8)
                plt.axis('off')
            except Exception as e:
                print(f"Error displaying image {img_path}: {e}")
        plt.suptitle(f"Cluster {cluster_id} Images")
        plt.savefig(os.path.join(output_folder, f'cluster_{cluster_id}_preview.png'))
        plt.close()

display_cluster_images(cluster_folders, new_image_paths)

# Interactive prompt to select which clusters to keep
print("\nCluster Previews Saved in Output Folder.")
print(f"Available clusters: {list(cluster_folders.keys())}")
while True:
    try:
        input_str = input("Enter cluster numbers to keep (e.g., 1,2,3): ")
        selected_clusters = [int(x) for x in input_str.replace(" ", "").split(",")]
        if all(c in cluster_folders for c in selected_clusters):
            break
        else:
            print(f"Invalid cluster numbers. Please choose from {list(cluster_folders.keys())}")
    except ValueError:
        print("Please enter valid integers separated by commas.")

# Keep only the selected clusters' folders
for cluster_id, folder in list(cluster_folders.items()):
    if cluster_id not in selected_clusters:
        try:
            shutil.rmtree(folder)
            print(f"Removed cluster_{cluster_id} folder")
        except Exception as e:
            print(f"Error removing cluster_{cluster_id} folder: {e}")
    else:
        print(f"Kept cluster_{cluster_id} folder at {folder}")
