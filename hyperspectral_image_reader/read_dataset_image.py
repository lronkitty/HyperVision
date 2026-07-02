import os
import sys
import argparse
import numpy as np

# Ensure workspace is in python path to allow importing configs
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from hyperspectral_pipelines import LoadHyperspectralImage
except ImportError as e:
    print(f"Error: Could not import LoadHyperspectralImage. Detail: {e}")
    sys.exit(1)

DATASET_EXTENSIONS = {
    # .mat / .h5 formats
    'harvard': '.mat', 'umld2015': '.mat', 'umns2002': '.mat', 'umns2004': '.mat', 
    'umos': '.mat', 'umri2015': '.mat', 'umemm': '.mat', 'hyperblood': '.mat',
    'arad_1k_31': '.mat', 'arad_1k_16': '.mat', 'cave': '.mat', 'fiftyoutdoor': '.mat',
    'icvl': '.h5', 'hs_sod': '.h5', 'hsodbitv2': '.mat',
    # .npy / .npz formats
    'hsidrive20': '.npy', 'aphid': '.npy',
    'hyperdrive': '.npz', 'hyperdrivevnir': '.npz', 'hyperdriveswir': '.npz',
    # ENVI formats (requires .hdr + raw file, pass the .hdr file path)
    'libhsi': '.hdr', 'virginia_tech_tree': '.hdr', 'vnihdhiatlimafb': '.hdr',
    # ENVI formats (with .bin, requires .hdr + .bin, pass the .bin file path)
    'deephsnir': '.bin', 'deephsvis': '.bin', 'deephsviscor': '.bin',
    # Image formats
    'hotvis': '.png', 'hotnir': '.png', 'hotrednir': '.png',
    'hsiroad': '.tif',
    # Custom format
    'hyperspectralcityv2': '.hsd',
}

def load_hypervision_matrix(file_path, dataset_name):
    """
    Loads a hyperspectral image and processes it into the exact matrix shape and scale
    expected by the HyperVision / HyperFree models.
    
    Args:
        file_path (str): Path to the image file.
        dataset_name (str): Name of the dataset (e.g., 'harvard', 'arad_1k_31', etc.).
        
    Returns:
        np.ndarray: Processed matrix of shape (H_ori, W_ori, C_hsi) scaled to [0, 255].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image path not found: {file_path}")

    # Validate file extension
    _, ext = os.path.splitext(file_path)
    expected_ext = DATASET_EXTENSIONS.get(dataset_name)
    if expected_ext and ext.lower() != expected_ext.lower():
        print(f"Warning: Expected file extension '{expected_ext}' for dataset '{dataset_name}', but got '{ext}'.")
        if expected_ext == '.hdr':
            print("Note: ENVI datasets require both the header (.hdr) and the raw binary data file. Please pass the path to the .hdr file.")
        elif expected_ext == '.bin':
            print("Note: DeepHS datasets require both the binary data (.bin) and the header (.hdr) file. Please pass the path to the .bin file.")
        elif expected_ext in ['.mat', '.h5']:
            print("Note: This dataset requires a MATLAB (.mat) or HDF5 (.h5) formatted cube.")
        elif expected_ext == '.npz':
            print("Note: This dataset requires a NumPy compressed archive (.npz) containing 'cube.npy'.")
        print()
        
    # Initialize the dataset loader pipeline
    loader = LoadHyperspectralImage(dataset_type=dataset_name, to_float32=True, append_rgb=True)
    
    # Run the transform
    results = {'img_path': file_path}
    results = loader(results)
    
    img = results['img']  # Loaded image of shape (H, W, C)
    
    # Check if the loaded image contains appended RGB channels.
    # The cache image might contain RGB (C = bands + 3), while the raw HSI might not (C = bands).
    num_hsi_channels = loader.bands
    actual_channels = img.shape[2]
    
    if actual_channels > num_hsi_channels:
        # Strip the last 3 channels (the appended RGB bands)
        img = img[:, :, :num_hsi_channels]

    # Min-Max normalization per image sample to [0, 255]
    hsi_min = img.min()
    hsi_max = img.max()
    if hsi_max > hsi_min:
        processed_matrix = 255.0 * (img - hsi_min) / (hsi_max - hsi_min)
    else:
        processed_matrix = np.zeros_like(img)
        
    return processed_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read HSI dataset image and output the matrix processed for HyperVision.")
    parser.add_argument('--path', type=str, required=True, help="Path to the HSI image file.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g. harvard, arad_1k_31, icvl, etc.).")
    parser.add_argument('--output', type=str, default=None, help="Optional path to save the output matrix as a .npy file.")
    
    args = parser.parse_args()
    
    try:
        matrix = load_hypervision_matrix(args.path, args.dataset)
        print("\nSuccessfully loaded and processed HSI image.")
        print(f"Matrix shape (H, W, C): {matrix.shape}")
        print(f"Value range: [{matrix.min():.2f}, {matrix.max():.2f}]")
        print(f"Data type: {matrix.dtype}")
        
        if args.output:
            np.save(args.output, matrix)
            print(f"Saved processed matrix to: {args.output}")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)
