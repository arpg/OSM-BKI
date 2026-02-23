import numpy as np
import argparse
import os

# SemanticKITTI to MCD Mapping
# Source: Derived from test_idea.py mapping logic
KITTI_TO_MCD = {
    40: 16, # Road -> Road
    44: 13, # Parking -> Parkinglot
    48: 18, # Sidewalk -> Sidewalk
    49: 18, # Other-ground -> Sidewalk (approx)
    70: 25, # Vegetation -> Vegetation
    71: 24, # Trunk -> Treetrunk
    72: 25, # Terrain -> Vegetation (approx)
    50: 2,  # Building -> Building
    51: 7,  # Fence -> Fence
    52: 20, # Other-structure -> Structure-other
    10: 26, 11: 1, 13: 26, 15: 26, 16: 26, 18: 26, 20: 27, # Vehicles -> Dyn/Bike/Other
    30: 14, 31: 1, 32: 26, # Person -> Ped, Bicyclist -> Bike, Motorcyclist -> Dyn
    80: 15, # Pole -> Pole
    81: 22, # Traffic-sign -> Traffic-sign
    99: 12, # Other-object -> Other
    0: 0,   # Unlabeled -> Barrier? Or keep 0 if 0 exists in MCD (Barrier is 0)
    1: 12   # Outlier -> Other
}

def convert_kitti_to_mcd(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    print(f"Loading SemanticKITTI labels from {input_path}...")
    
    # Load raw data (assuming .label or .bin format with uint32)
    # Lower 16 bits = semantic label, Upper 16 bits = instance id
    raw_data = np.fromfile(input_path, dtype=np.uint32)
    
    sem_labels = raw_data & 0xFFFF
    instance_ids = raw_data & 0xFFFF0000
    
    # Create output array initialized to 'Other' (12) or 0 (Barrier) for unknown classes
    # Using 12 (Other) as safe default for unmatched classes
    new_sem_labels = np.full_like(sem_labels, 12) 
    
    # Apply mapping
    mapped_count = 0
    for k_id, m_id in KITTI_TO_MCD.items():
        mask = (sem_labels == k_id)
        count = np.sum(mask)
        if count > 0:
            new_sem_labels[mask] = m_id
            mapped_count += count
            
    # Check for unmapped labels
    unique_unmapped = np.unique(sem_labels[new_sem_labels == 12])
    # Filter out ones that were explicitly mapped to 12
    explicit_other = [k for k, v in KITTI_TO_MCD.items() if v == 12]
    really_unmapped = [u for u in unique_unmapped if u not in explicit_other]
    
    if really_unmapped:
        print(f"Warning: The following SemanticKITTI classes were mapped to 'Other' (12) by default: {really_unmapped}")

    # Recombine with instance IDs (preserving original instances)
    final_data = instance_ids | new_sem_labels.astype(np.uint32)
    
    print(f"Saving MCD labels to {output_path}...")
    final_data.tofile(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SemanticKITTI label file(s) to MCD label file(s)")
    
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to input SemanticKITTI .label/.bin file OR directory")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to save output MCD .label/.bin file OR directory")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        elif not os.path.isdir(args.output):
            raise NotADirectoryError(f"Output must be a directory if input is a directory: {args.output}")
            
        import glob
        input_files = sorted(glob.glob(os.path.join(args.input, "*.label")))
        if not input_files:
             input_files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
             
        print(f"Found {len(input_files)} files in {args.input}")
        
        for in_file in input_files:
            filename = os.path.basename(in_file)
            out_file = os.path.join(args.output, filename)
            convert_kitti_to_mcd(in_file, out_file)
            
        print("Batch conversion done.")
    else:
        # Single file mode
        convert_kitti_to_mcd(args.input, args.output)
        print("Done.")
