import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict
import math
import os
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import seaborn as sns

try:
    import utils as ut # Tries to import utils for get_model_input_channels
    HAS_UTILS = True
except ImportError:
    print("WARNING: Could not import 'utils.py'. Channel logic may be limited.")
    HAS_UTILS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _preprocess_for_model(img_tensor, model):
    """
    Helper function to preprocess ONE image for prediction.
    This function is crucial and must be consistent.
    """
    input_tensor = img_tensor.clone()
    # Assumes that for SOTA models, the input will be 3 channels, 224x224
    # and for custom models, it will be the native size.
    model_type_str = str(type(model))
    is_sota_model = 'resnet' in model_type_str.lower() or 'mobile' in model_type_str.lower() or 'vgg' in model_type_str.lower()

    # Ensures 3 channels for SOTA models
    if is_sota_model and input_tensor.shape[0] == 1:
        input_tensor = input_tensor.repeat(3, 1, 1)

    # Resizes for SOTA models
    if is_sota_model and (input_tensor.shape[1:] != (224, 224)):
        resize_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
        input_tensor = resize_transform(input_tensor)
        
    return input_tensor

def select_correctly_classified_images(model, dataset, num_per_class=1):
    """
    Selects up to `num_per_class` CORRECTLY CLASSIFIED images
    per class, processing 1 by 1 (shuffled) and stopping early.
    """
    model.eval()
    counts = defaultdict(int)
    selected = []
    
    try:
        num_target_classes = len(dataset.classes)
        all_possible_labels = list(range(num_target_classes))
    except AttributeError:
        print("WARNING: Could not determine the exact number of classes. Assuming 10.")
        num_target_classes = 10 
        all_possible_labels = list(range(10))

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    progress_bar = tqdm(indices, desc=f"Searching for {num_per_class} img/class", unit="img", dynamic_ncols=True)

    for idx in progress_bar:
        img_tensor, label = dataset[idx] 
        label_item = label if isinstance(label, int) else label.item()

        if counts[label_item] >= num_per_class:
            continue

        if not isinstance(img_tensor, torch.Tensor):
             img_tensor = transforms.ToTensor()(img_tensor).float()

        input_for_pred = _preprocess_for_model(img_tensor.clone(), model)
        if input_for_pred is None: continue 

        with torch.no_grad():
            pred = model(input_for_pred.to(DEVICE).unsqueeze(0)).argmax(dim=1).item()

        if pred == label_item:
            selected.append((img_tensor.cpu(), label_item, idx, pred)) 
            counts[label_item] += 1
            # Update information on the progress bar
            progress_bar.set_postfix(found=len(selected), classes=f"{len(counts)}/{num_target_classes}")
        
        num_classes_completed = sum(1 for class_idx in all_possible_labels if counts[class_idx] >= num_per_class)
        
        if num_classes_completed >= num_target_classes:
            progress_bar.close()
            print(f"\n  -> Reached the desired number for all classes ({len(selected)} images). Stopping selection.")
            break 
            
    if not selected:
        print("!!! ATTENTION: No images selected.")
    else:
        print(f"Selection complete. {len(selected)} images selected.")
        for lbl in sorted(counts.keys()):
            print(f"  Class {lbl}: {counts[lbl]} images selected.")
            
    return selected

def select_one_per_class(dataset, num_per_class=3):
    counts = defaultdict(int)
    selected = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if isinstance(img, np.ndarray): img = torch.from_numpy(img).float()
        elif not isinstance(img, torch.Tensor): img = transforms.ToTensor()(img).float()
        if img.shape[0] == 1 and img.shape[0] != 3: img = img.repeat(3, 1, 1)
        if img.shape[1] != 28 and img.shape[2] != 28:
            img = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])(img)
        if isinstance(label, torch.Tensor): label = label.item()
        if counts[label] < num_per_class:
            selected.append((img, label, idx))
            counts[label] += 1
        if len(counts) >= 10 and all(c >= num_per_class for c in counts.values()): break
    return selected

def _recursive_scaled_zz(model, low_res_img, orig_pred, idx, label, max_level, records, x0, y0, w, h, level):
    """Recursive helper function for Scaled Zero Zones."""
    
    if w < 4 or h < 4 or level > max_level:
        return

    zone_coords = [
        (x0, y0, w // 2, h // 2),              # Top-Left
        (x0 + w // 2, y0, w - w // 2, h // 2),  # Top-Right
        (x0, y0 + h // 2, w // 2, h - h // 2),  # Bottom-Left
        (x0 + w // 2, y0 + h // 2, w - w // 2, h - h // 2) # Bottom-Right
    ]

    for zx, zy, zw, zh in zone_coords:
        if zw <= 0 or zh <= 0:
            continue

        # 1. Perturb the low-resolution image
        perturbed_low_res = low_res_img.clone()
        perturbed_low_res[:, zy:zy+zh, zx:zx+zw] = 0.0

        # 2. Preprocess the perturbed image (which includes resizing to 224x224)
        input_for_model = _preprocess_for_model(perturbed_low_res, model)

        # 3. Re-evaluate the model
        with torch.no_grad():
            new_pred = model(input_for_model.to(DEVICE).unsqueeze(0)).argmax().item()

        # 4. Record and, if the prediction changed, continue the recursion
        changed = (new_pred != orig_pred)
        records.append({
            'image_id': idx, 'true_label': label, 'orig_pred': orig_pred,
            'zone_coords': f"[{zy}..{zy+zh-1}][{zx}..{zx+zw-1}]",
            'zone_pred': new_pred, 'changed': changed, 'level': level
        })
        
        if changed:
            _recursive_scaled_zz(model, low_res_img, orig_pred, idx, label, max_level, records, zx, zy, zw, zh, level + 1)

def generate_scaled_zero_zone_analysis(model, dataset, run_dir: Path, num_images_per_class=1, max_level=1000):
    """
    Runs the "Scaled Zero Zones" analysis and saves the raw heatmaps
    to .npy files, grouped by class.
    """
    model.eval()
    print(f"\n--- Starting Scaled Zero Zones Analysis (Max Level: {max_level}) ---")
    
    selected_imgs = select_correctly_classified_images(model, dataset, num_images_per_class)
    if not selected_imgs:
        print("No CORRECT images selected for analysis.")
        return

    # Dictionary to group heatmaps by class before saving
    heatmaps_by_class = defaultdict(list)

    for img_tensor, label, idx, orig_pred in tqdm(selected_imgs, desc="Generating Scaled ZZ Heatmaps", unit="img"):
        
        img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
        records = []

        # Start recursion on the low-resolution image
        _recursive_scaled_zz(model, img_tensor.to(DEVICE), orig_pred, idx, label, max_level, records,
                             x0=0, y0=0, w=img_w, h=img_h, level=1)
        
        # Generate the numeric heatmap from the records
        individual_heatmap = np.zeros((img_h, img_w), dtype=np.float32) # Using float32 is efficient
        for rec in records:
            if rec['changed']:
                try:
                    parts = rec['zone_coords'].replace('[', '').split(']')
                    rows_part, cols_part = parts[0], parts[1]
                    y0, y1 = map(int, rows_part.split('..'))
                    x0, x1 = map(int, cols_part.split('..'))
                    individual_heatmap[y0:y1+1, x0:x1+1] += 1
                except:
                    continue 

        # Add the generated heatmap to its class list
        heatmaps_by_class[label].append(individual_heatmap)

    # --- .NPY SAVING LOGIC ---
    npy_output_dir = run_dir / "individual_heatmaps_npy"
    npy_output_dir.mkdir(exist_ok=True)
    print(f"\nSaving individual heatmaps into .npy arrays at: {npy_output_dir}")

    for label, heatmap_list in heatmaps_by_class.items():
        if not heatmap_list:
            print(f"  -> Class {label}: No heatmap generated.")
            continue
        
        # Stack the list of 2D arrays into a single 3D array
        stacked_heatmaps = np.stack(heatmap_list, axis=0)
        
        save_path = npy_output_dir / f"heatmaps_class_{label}.npy"
        np.save(save_path, stacked_heatmaps)
        print(f"  -> Class {label}: Saved array with shape {stacked_heatmaps.shape} at {save_path.name}")

    print("\nScaled Zero Zones analysis (generation of .npy) completed.")
    print(f"Suggested next step: run the 'aggregate_only' or 'pixel_attack' analysis using the directory '{npy_output_dir}'")

def recursive_zero_zones(model, img, orig_pred, img_id, label, run_dir, x0, x1, y0, y1, level, max_level, records):
    H_zone_full, W_zone_full = y1 - y0, x1 - x0
    orig_img_h, orig_img_w = img.shape[1], img.shape[2]
    if H_zone_full < 4 or W_zone_full < 4 or level > max_level: return
    zone_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    h_half, w_half = H_zone_full // 2, W_zone_full // 2
    if h_half == 0 or w_half == 0: return
    for zone_name_str in zone_names:
        if zone_name_str == 'top_left': sx0, sx1_excl, sy0, sy1_excl = x0, x0 + w_half, y0, y0 + h_half
        elif zone_name_str == 'top_right': sx0, sx1_excl, sy0, sy1_excl = x0 + w_half, x1, y0, y0 + h_half
        elif zone_name_str == 'bottom_left': sx0, sx1_excl, sy0, sy1_excl = x0, x0 + w_half, y0 + h_half, y1
        else: sx0, sx1_excl, sy0, sy1_excl = x0 + w_half, x1, y0 + h_half, y1
        current_zone_h, current_zone_w = sy1_excl - sy0, sx1_excl - sx0
        if current_zone_w <= 0 or current_zone_h <= 0: continue
        img_z = img.clone(); img_z[:, sy0:sy1_excl, sx0:sx1_excl] = 0
        zone_coords_display_str = f"[{sy0}..{sy1_excl-1}][{sx0}..{sx1_excl-1}]"
        input_tensor_for_pred = _preprocess_for_model(img_z, model)
        with torch.no_grad(): out = model(input_tensor_for_pred.to(DEVICE).unsqueeze(0))
        zone_pred_val = out.argmax(dim=1).item()
        changed_pred_flag = (zone_pred_val != orig_pred)
        records.append({'image_id': img_id, 'true_label': label, 'orig_pred': orig_pred, 'zone_name': zone_name_str, 'zone_coords': zone_coords_display_str, 'zone_pred': zone_pred_val, 'changed': changed_pred_flag, 'img_height': orig_img_h, 'img_width': orig_img_w, 'level': level, 'zone_area_pixels': current_zone_h * current_zone_w})
        if changed_pred_flag: recursive_zero_zones(model, img, orig_pred, img_id, label, run_dir, sx0, sx1_excl, sy0, sy1_excl, level + 1, max_level, records)

def parse_coords(coords_str):
    try:
        parts = coords_str.replace('[', '').split(']')
        rows_part, cols_part = parts[0], parts[1]
        r0, r1 = map(int, rows_part.split('..'))
        c0, c1 = map(int, cols_part.split('..'))
        return r0, r1, c0, c1
    except Exception as e:
        print(f"WARNING: Failed to parse coordinates: '{coords_str}'. Error: {e}. Returning 0s.")
        return 0, 0, 0, 0

def generate_heatmap_overlay(records, img_tensor, img_id, run_dir, orig_pred_for_title, show_matrix=False):
    orig_img_h, orig_img_w = img_tensor.shape[1], img_tensor.shape[2]
    heatmap = np.zeros((orig_img_h, orig_img_w), dtype=float)
    changed_zones_count = 0
    current_image_records = [rec for rec in records if rec['image_id'] == img_id]
    if not current_image_records:
        # print(f"Warning: No records for heatmap of image ID {img_id}.")
        return
    true_label_for_title = current_image_records[0]['true_label']
    for rec in current_image_records:
        if rec['changed']:
            r0, r1, c0, c1 = parse_coords(rec['zone_coords'])
            heatmap[r0:min(r1 + 1, orig_img_h), c0:min(c1 + 1, orig_img_w)] += 1
            changed_zones_count += 1
    
    if show_matrix:
        print(f"\n--- Numeric 'Zero Zones' Heatmap Matrix (ID: {img_id}) ---")
        print("Values represent the count of times a recursive zone caused a prediction change.")
        # For small matrices (like MNIST 28x28), printing is legible.
        # For larger ones, it may be truncated by NumPy.
        np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=0)
        print(heatmap)
        np.set_printoptions() 
        print("--- End of Matrix ---\n")        

    if heatmap.max() > 0: heatmap_norm = heatmap / heatmap.max()
    else: heatmap_norm = heatmap
    img_display_overlay = img_tensor.squeeze(0).cpu()
    img_display_overlay_clamped = torch.clamp(img_display_overlay, 0, 1)
    img_np = img_display_overlay_clamped.permute(1, 2, 0).numpy() if img_display_overlay_clamped.shape[0] == 3 else img_display_overlay_clamped.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np, cmap='gray' if img_tensor.shape[0] == 1 else None, interpolation='nearest')
    plt.imshow(heatmap_norm, cmap='jet', interpolation='nearest', alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(label='Prediction Change Frequency (Normalized)')
    title_heatmap = (f"Heatmap - ID:{img_id} (OrigRes:{orig_img_h}x{orig_img_w})\n"
                     f"TrueLabel:{true_label_for_title} OrigPred:{orig_pred_for_title} "
                     f"({changed_zones_count} zones changed prediction)")
    plt.title(title_heatmap, fontsize=10)
    plt.axis('off')
    heatmaps_dir = run_dir / "heatmaps_overlay"
    heatmaps_dir.mkdir(exist_ok=True)
    plt.savefig(heatmaps_dir / f"overlay_heatmap_id{img_id}_res{orig_img_h}x{orig_img_w}.png", bbox_inches='tight')
    plt.close()

def _compute_zero_zone_matrix(model, img_tensor, orig_pred, idx, label, max_level):
    """Refactored: Computes and returns the heatmap matrix for Zero Zones of ONE image."""
    img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
    records = []
    # The recursive function populates the 'records' list
    # We pass a dummy run_dir as we don't want to save individual recursion images here.
    dummy_run_dir = Path("./dummy_zz_temp")
    recursive_zero_zones(model, img_tensor.to(DEVICE), orig_pred, idx, label, dummy_run_dir, 0, img_w, 0, img_h, level=1, max_level=max_level, records=records)

    heatmap = np.zeros((img_h, img_w), dtype=float)
    for rec in records:
        if rec['changed']:
            r0, r1, c0, c1 = parse_coords(rec['zone_coords'])
            heatmap[r0:min(r1 + 1, img_h), c0:min(c1 + 1, img_w)] += 1
    return heatmap

def generate_zero_zone_analysis(model, dataset, run_dir: Path, num_images_per_class=1, max_level=1000, show_matrix=False):
    model.eval()
    selected_imgs = select_correctly_classified_images(model, dataset, num_images_per_class)
    if not selected_imgs:
        print("No CORRECT images selected for Zero Zones analysis.")
        return

    print(f"\nStarting Zero Zones analysis for {len(selected_imgs)} images...")
    
    # Dictionary to group heatmaps by class before saving
    heatmaps_by_class = defaultdict(list)

    for img_tensor, label, idx, orig_pred in tqdm(selected_imgs, desc="Generating Zero-Zone Heatmaps", unit="img"):
        # Generate the individual heatmap matrix WITHOUT saving intermediate recursion plots
        individual_heatmap = _compute_zero_zone_matrix(model, img_tensor, orig_pred, idx, label, max_level)
        
        # Add the heatmap to its class list
        heatmaps_by_class[label].append(individual_heatmap)

        # Plotting the individual heatmap (optional) can still be done here if desired
        # generate_heatmap_overlay(...)

    # --- .NPY SAVING LOGIC ---
    npy_output_dir = run_dir / "individual_heatmaps_npy"
    npy_output_dir.mkdir(exist_ok=True)
    print(f"\nSaving individual heatmaps in .npy arrays at: {npy_output_dir}")

    for label, heatmap_list in heatmaps_by_class.items():
        # Stack the list of 2D arrays into a single 3D array
        stacked_heatmaps = np.stack(heatmap_list, axis=0)
        
        save_path = npy_output_dir / f"heatmaps_class_{label}.npy"
        np.save(save_path, stacked_heatmaps)
        print(f"  -> Class {label}: Saved array with shape {stacked_heatmaps.shape} at {save_path.name}")
    
    print("\nZero Zones analysis (.npy generation) completed.")
    
def generate_knn_heatmap(model, dataset, run_dir: Path, num_images_per_class: int = 1, k_neighbors_to_paint: int = 10, num_key_pixels_to_evaluate: int = 200, perturb_patch_size: int = 5, show_matrix: bool = False):
    model.eval()
    selected_imgs_data = select_correctly_classified_images(model, dataset, num_images_per_class)
    if not selected_imgs_data:
        print("No CORRECTLY CLASSIFIED images selected for KNN heatmap analysis.")
        return
    knn_heatmaps_dir = run_dir / "knn_heatmaps_generated"
    knn_heatmaps_dir.mkdir(parents=True, exist_ok=True)
    base_images_dir = run_dir / "base_images_for_knn"
    base_images_dir.mkdir(parents=True, exist_ok=True)
    summary_records = []
    print(f"\nStarting KNN Heatmap generation (k={k_neighbors_to_paint}, test_pixels={num_key_pixels_to_evaluate})")

    for img_tensor, true_label, original_idx, orig_pred_class in tqdm(selected_imgs_data, desc="Generating KNN Heatmaps", unit="img"):
        img_tensor_device = img_tensor.clone().to(DEVICE)
        _, img_h, img_w = img_tensor_device.shape
        
        all_pixel_coords = np.array([(r, c) for r in range(img_h) for c in range(img_w)])
        knn_spatial_model = NearestNeighbors(n_neighbors=k_neighbors_to_paint, algorithm='ball_tree')
        knn_spatial_model.fit(all_pixel_coords)
        
        num_steps_h = int(np.sqrt(num_key_pixels_to_evaluate * img_h / img_w))
        num_steps_w = int(np.sqrt(num_key_pixels_to_evaluate * img_w / img_h))
        if num_steps_h == 0: num_steps_h = 1
        if num_steps_w == 0: num_steps_w = 1
        h_indices = np.linspace(0, img_h - 1, num_steps_h, dtype=int)
        w_indices = np.linspace(0, img_w - 1, num_steps_w, dtype=int)
        key_pixels_to_test = [(r_idx, c_idx) for r_idx in h_indices for c_idx in w_indices]
        if not key_pixels_to_test: key_pixels_to_test = [(np.random.randint(img_h), np.random.randint(img_w)) for _ in range(min(10, img_h * img_w))]

        current_image_heatmap = np.zeros((img_h, img_w), dtype=float)
        impactful_pixels_found_count = 0

        pixel_iterator = tqdm(key_pixels_to_test, desc=f"  Analyzing pixels ID {original_idx}", leave=False, unit="pixel", dynamic_ncols=True)
        for r_key, c_key in pixel_iterator:
            img_perturbed = img_tensor_device.clone()
            r_start, r_end = max(0, r_key - perturb_patch_size // 2), min(img_h, r_key + (perturb_patch_size // 2) + (perturb_patch_size % 2))
            c_start, c_end = max(0, c_key - perturb_patch_size // 2), min(img_w, c_key + (perturb_patch_size // 2) + (perturb_patch_size % 2))
            img_perturbed[:, r_start:r_end, c_start:c_end] = 0.0
            input_for_pred = _preprocess_for_model(img_perturbed.clone(), model)

            with torch.no_grad():
                output_perturbed = model(input_for_pred.unsqueeze(0))
                pred_perturbed_class = output_perturbed.argmax(dim=1).item()

            if pred_perturbed_class != orig_pred_class:
                impactful_pixels_found_count += 1
                _, neighbor_indices_1d = knn_spatial_model.kneighbors(np.array([[r_key, c_key]]))
                for neighbor_1d_idx in neighbor_indices_1d[0]:
                    neighbor_r, neighbor_c = all_pixel_coords[neighbor_1d_idx]
                    current_image_heatmap[neighbor_r, neighbor_c] += 1
        
        if show_matrix:
            print(f"\n--- Numeric 'KNN' Heatmap Matrix (ID: {original_idx}) ---")
            print("Values represent the 'vote' count from pixels neighboring an impact point.")
            np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=0)
            print(current_image_heatmap)
            np.set_printoptions() 
            print("--- End of Matrix ---\n")
        
        summary_records.append({'image_id': original_idx, 'true_label': true_label, 'original_prediction': orig_pred_class, 'num_key_pixels_evaluated': len(key_pixels_to_test), 'num_impactful_key_pixels': impactful_pixels_found_count, 'prediction_changed_by_perturbation': impactful_pixels_found_count > 0})
        
        if current_image_heatmap.max() > 0: heatmap_normalized = current_image_heatmap / current_image_heatmap.max()
        else: heatmap_normalized = current_image_heatmap

        display_original_img_np = img_tensor.cpu().permute(1, 2, 0).numpy() if img_tensor.shape[0] == 3 else img_tensor.cpu().squeeze(0).numpy()
        display_original_img_np = np.clip(display_original_img_np, 0, 1)

        plt.figure(figsize=(8, 8))
        plt.imshow(display_original_img_np, cmap='gray' if img_tensor.shape[0] == 1 else None, interpolation='nearest')
        plt.imshow(heatmap_normalized, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        plt.colorbar(label=f'KNN Impact (k={k_neighbors_to_paint})')
        title_str = (f"KNN Heatmap - ID:{original_idx}\n"
                     f"TrueLabel:{true_label}, OrigPred:{orig_pred_class}\n"
                     f"Perturb Patch: {perturb_patch_size}x{perturb_patch_size}, Evaluated Pixels: {len(key_pixels_to_test)}")
        plt.title(title_str, fontsize=10)
        plt.axis('off')
        heatmap_save_path = knn_heatmaps_dir / f"knn_heatmap_id{original_idx}_k{k_neighbors_to_paint}.png"
        plt.savefig(heatmap_save_path, bbox_inches='tight')
        plt.close()

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        summary_csv_path = run_dir / "knn_analysis_summary.csv"
        try:
            df_summary.to_csv(summary_csv_path, index=False)
            print(f"KNN Heatmap analysis summary saved at: {summary_csv_path}")
        except Exception as e:
            print(f"Error saving KNN analysis CSV summary: {e}")
    else:
        print("No summary data to save for KNN analysis.")
    print(f"KNN Heatmap analysis completed. Results in: {run_dir}")
    
    
def aggregate_and_plot_from_npy(npy_input_dir: Path, output_dir: Path):
    """
    Reads .npy files containing heatmaps, aggregates them by class, and saves the final plots.
    This is an independent function for Step 2.
    """
    print(f"\n--- Starting Step 2: Aggregation from .npy files ---")
    print(f"Reading files from: {npy_input_dir}")
    print(f"Saving aggregated plots to: {output_dir}")
    
    output_dir.mkdir(exist_ok=True)
    
    npy_files = list(npy_input_dir.glob("heatmaps_class_*.npy"))
    if not npy_files:
        print(f"WARNING: No 'heatmaps_class_*.npy' files found in {npy_input_dir}")
        return

    for file_path in tqdm(npy_files, desc="Processing classes", unit="file"):
        try:
            # Extracts the class label from the filename
            label = int(file_path.stem.split('_')[-1])
        except (ValueError, IndexError):
            print(f"WARNING: Could not extract label from filename {file_path.name}. Skipping.")
            continue

        # Loads the heatmap array
        stacked_heatmaps = np.load(file_path)
        num_aggregated = stacked_heatmaps.shape[0]

        # Calculates the average heatmap
        average_heatmap = np.mean(stacked_heatmaps, axis=0)

        # Normalizes for visualization
        if average_heatmap.max() > 0:
            viz_heatmap = average_heatmap / average_heatmap.max()
        else:
            viz_heatmap = average_heatmap

        print(f"Class {label}: {num_aggregated} heatmaps aggregated. Max avg value: {average_heatmap.max():.2f}")

        # Plots and saves the final result
        plt.figure(figsize=(6, 6))
        plt.imshow(viz_heatmap, cmap='jet', interpolation='nearest')
        plt.colorbar(label='Normalized Average Importance')
        plt.title(f"Aggregated Heatmap - Class {label}\n({num_aggregated} images)")
        plt.axis('off')
        
        img_path = output_dir / f"aggregated_class_{label}.png"
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

    print(f"\nAggregation and plotting completed. Results in {output_dir}.")    
    
# --- NEW FEATURE: Heatmap-Guided Pixel Attack ---

# --- UPDATED Pixel Attack Functions ---

def _run_single_pixel_attack(model, image_tensor, true_label, attack_heatmap, attack_patch_size=1):
    """
    Executes the perturbation attack, zeroing out patches in order of importance
    from the heatmap until the prediction changes.

    Returns:
         - pixels_changed (int): Total number of unique modified pixels.
         - final_prediction (int): The new (incorrect) prediction.
         - perturbed_image (Tensor): The image with zeroed-out patches.
    """
    # 1. Get the order of center pixels to be attacked
    flat_heatmap = attack_heatmap.flatten()
    sorted_pixel_indices = np.argsort(flat_heatmap)[::-1]
    
    img_h, img_w = image_tensor.shape[1], image_tensor.shape[2]
    
    # Mask to track already zeroed pixels
    occlusion_mask = torch.zeros((img_h, img_w), dtype=torch.bool, device=image_tensor.device)

    # 2. Iterate and attack with patches
    for i, flat_idx in enumerate(sorted_pixel_indices):
        # Convert the flat index back to center pixel coordinates
        center_row, center_col = np.unravel_index(flat_idx, attack_heatmap.shape)
        
        # Define the coordinates of the patch to be zeroed
        r_start = max(0, center_row - attack_patch_size // 2)
        r_end = min(img_h, center_row + attack_patch_size // 2 + 1)
        c_start = max(0, center_col - attack_patch_size // 2)
        c_end = min(img_w, center_col + attack_patch_size // 2 + 1)
        
        # Update the occlusion mask
        occlusion_mask[r_start:r_end, c_start:c_end] = True
        
        # Create the perturbed image by applying the mask
        # Multiplying by the inverted mask zeros out the pixels where the mask is True
        perturbed_image = image_tensor.clone() * (~occlusion_mask)
        
        # 3. Re-evaluate the model with the perturbed image
        with torch.no_grad():
            input_for_pred = _preprocess_for_model(perturbed_image.clone(), model)
            new_pred = model(input_for_pred.to(DEVICE).unsqueeze(0)).argmax().item()
            
        # 4. Check if the attack was successful
        if new_pred != true_label:
            pixels_changed = occlusion_mask.sum().item()
            return pixels_changed, new_pred, perturbed_image
            
    # If the loop finishes, the attack failed to change the prediction
    pixels_changed = occlusion_mask.sum().item()
    return pixels_changed, true_label, perturbed_image


def generate_pixel_attack_report(model, dataset, aggregated_heatmaps_dir: Path, run_dir: Path, num_images_per_class: int, attack_patch_size: int = 1):
    """
    Orchestrates the pixel attack test (with patches) and generates a report.
    """
    print(f"--- Starting Pixel Attack Analysis (Patch: {attack_patch_size}x{attack_patch_size}) ---")
    
    # 1. Load the aggregated heatmaps
    aggregated_heatmaps = {}
    npy_files = list(aggregated_heatmaps_dir.glob("*.npy"))
    if not npy_files:
        print(f"ERROR: No .npy heatmap files found in '{aggregated_heatmaps_dir}'.")
        print("First run an analysis that generates .npy (e.g., scaled_zero_zone).")
        return
        
    for file_path in npy_files:
        try:
            label = int(file_path.stem.split('_')[-1])
            aggregated_heatmaps[label] = np.load(file_path)
        except Exception as e:
            print(f"WARNING: Failed to load or parse {file_path.name}: {e}")
            continue
    print(f"Loaded {len(aggregated_heatmaps)} aggregated heatmaps from {aggregated_heatmaps_dir}.")

    # 2. Select test images (that the model gets right)
    test_images = select_correctly_classified_images(model, dataset, num_images_per_class)
    if not test_images:
        print("No correctly classified images found to attack.")
        return

    attack_results = []
    output_plots_dir = run_dir / "pixel_attack_plots"
    output_plots_dir.mkdir(exist_ok=True)

    # 3. Execute the attack for each test image
    for img_tensor, true_label, idx, _ in tqdm(test_images, desc="Executing Patch Attacks"):
        if true_label not in aggregated_heatmaps:
            print(f"WARNING: No aggregated heatmap for class {true_label}. Skipping image {idx}.")
            continue

        # Uses the average of the heatmaps for the class as a guide
        attack_heatmap = np.mean(aggregated_heatmaps[true_label], axis=0)

        # Run the attack with patches
        pixels_changed, final_pred, perturbed_img = _run_single_pixel_attack(
            model, img_tensor.to(DEVICE), true_label, attack_heatmap, attack_patch_size
        )
        
        total_pixels = img_tensor.shape[1] * img_tensor.shape[2]
        attack_results.append({
            "image_id": idx,
            "true_label": true_label,
            "final_prediction": final_pred,
            "pixels_changed": pixels_changed,
            "pixels_changed_percent": (pixels_changed / total_pixels) * 100
        })

        # 4. Generate a visual result plot for this attack
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        img_tensor_cpu = img_tensor.cpu()
        perturbed_img_cpu = perturbed_img.cpu()
        
        axes[0].imshow(img_tensor_cpu.permute(1, 2, 0).squeeze(), cmap='gray')
        axes[0].set_title(f"Original (ID: {idx})\nPred: {true_label}")
        axes[0].axis('off')
        
        mask = (img_tensor_cpu - perturbed_img_cpu).abs().permute(1, 2, 0).squeeze()
        axes[1].imshow(mask, cmap='hot')
        axes[1].set_title(f"{pixels_changed} Altered Pixels")
        axes[1].axis('off')

        axes[2].imshow(perturbed_img_cpu.permute(1, 2, 0).squeeze(), cmap='gray')
        axes[2].set_title(f"Attacked\nNew Pred: {final_pred}")
        axes[2].axis('off')

        fig.suptitle(f"Attack Result (Patch {attack_patch_size}x{attack_patch_size}) - Class {true_label} -> {final_pred}", fontsize=16)
        plt.savefig(output_plots_dir / f"attack_id_{idx}_class_{true_label}.png", bbox_inches='tight')
        plt.close(fig)

    # 5. Present a final report
    if attack_results:
        df_results = pd.DataFrame(attack_results)
        df_results = df_results.sort_values(by="pixels_changed_percent")
        
        print("\n--- Final Pixel Attack Report ---")
        print(df_results.to_string(index=False))
        
        df_results.to_csv(run_dir / "pixel_attack_summary.csv", index=False)
        
        avg_pixels_changed = df_results["pixels_changed_percent"].mean()
        print(f"\nSummary: On average, {avg_pixels_changed:.2f}% of the pixels needed to be altered to fool the model.")
    
    print(f"Attack plots saved in: {output_plots_dir}")    
    
def _analyze_attack_report_plots(df: pd.DataFrame, output_dir: Path, class_names: dict = None):
    """Generates all analysis plots, using class names if provided."""
    print("\n--- Generating Attack Report Analysis Plots ---")
    output_dir.mkdir(exist_ok=True)
    
    plot_df = df.copy()

    # Only map if it's still an int
    if class_names and plot_df['true_label'].dtype != object:
        print(" -> Mapping class indices to names for plots.")
        plot_df['true_label'] = plot_df['true_label'].map(class_names)
        plot_df['final_prediction'] = plot_df['final_prediction'].map(class_names)
    else:
        print(" -> Labels are already strings. No mapping will be performed.")
    
    # 1. Confusion Matrix
    successful_attacks = plot_df[plot_df['true_label'] != plot_df['final_prediction']]
    if not successful_attacks.empty:
        all_labels = sorted(list(class_names.values())) if class_names else sorted(df['true_label'].unique())
        confusion_matrix_counts = pd.crosstab(
            successful_attacks['true_label'], successful_attacks['final_prediction']
        ).reindex(index=all_labels, columns=all_labels, fill_value=0)
        
        plt.figure(figsize=(14, 11)); 
        sns.heatmap(confusion_matrix_counts, annot=True, cmap="YlGnBu", fmt='d'); 
        plt.title("Attack Confusion Matrix (Count)")
        plt.ylabel("True Class"); plt.xlabel("Predicted Class (After Attack)")
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "attack_confusion_matrix.png", bbox_inches='tight'); 
        plt.close()
        print(f" -> Confusion Matrix plot saved.")

    # 2. Robustness per Class
    all_class_labels = sorted(list(class_names.values())) if class_names else sorted(df['true_label'].unique())
    avg_pixels_per_class = plot_df.groupby('true_label')['pixels_changed_percent'].mean()

    # Fill with 100.0 for missing classes (not successfully attacked)
    for c in all_class_labels:
        if c not in avg_pixels_per_class:
            avg_pixels_per_class[c] = 100.0

    avg_pixels_per_class = avg_pixels_per_class.reindex(all_class_labels) # sort by original label

    plt.figure(figsize=(14, 7)); 
    avg_pixels_per_class.plot(kind='bar', color='skyblue'); 
    plt.title("Average Robustness per Class"); plt.ylabel("Average % of Modified Pixels"); plt.xlabel("Class"); 
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.grid(axis='y', linestyle='--'); 
    plt.savefig(output_dir / "average_robustness_per_class.png", bbox_inches='tight'); plt.close()
    print(f" -> Robustness per Class plot saved.")

    # 3. Robustness Distribution
    classes_in_df = plot_df['true_label'].unique()
    missing_classes = [c for c in all_class_labels if c not in classes_in_df]

    # Add "fake" entries with 100% for unattacked classes (optional for boxplot)
    for c in missing_classes:
        plot_df = pd.concat([plot_df, pd.DataFrame([{
            'true_label': c,
            'pixels_changed_percent': 100.0,
            'final_prediction': c  # dummy
        }])], ignore_index=True)      
    
    plt.figure(figsize=(16, 8)); 
    sns.boxplot(data=plot_df, x='true_label', y='pixels_changed_percent', palette='viridis', order=avg_pixels_per_class.index); 
    plt.title("Robustness Distribution per Class"); plt.ylabel("% of Modified Pixels"); plt.xlabel("Class"); 
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.grid(axis='y', linestyle='--'); 
    plt.savefig(output_dir / "robustness_distribution_boxplot.png", bbox_inches='tight'); plt.close()
    print(f" -> Robustness Distribution plot saved.")


import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt

def _generate_gradcam_heatmaps_in_memory(model, images_to_process, target_layer):
    """Generates Grad-CAM heatmaps for a list of images and returns them in memory."""
    model.eval()

    heatmaps_by_class = defaultdict(list)

    # Temporarily store activations and gradients
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Register the hooks
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)  # more stable for simple models

    for img_tensor, label, idx, orig_pred in tqdm(images_to_process, desc="Generating Heatmaps (Grad-CAM)", unit="img"):
        # Reset for each image
        activations = None
        gradients = None

        img_for_pred = _preprocess_for_model(img_tensor.clone(), model).to(DEVICE).unsqueeze(0)
        img_for_pred.requires_grad_(True)

        # Forward
        output = model(img_for_pred)

        # Backward for the predicted class
        model.zero_grad()
        score = output[0, orig_pred]
        score.backward()

        if gradients is None or activations is None:
            print(f"WARNING: Failed to capture gradients or activations for image {idx}. Skipping.")
            continue

        # Average gradients per channel
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # (C,)

        # Apply weights to activations (without overwriting original)
        weighted_activations = activations * pooled_gradients[None, :, None, None]  # broadcasting

        # Calculate raw heatmap
        heatmap = torch.mean(weighted_activations, dim=1).squeeze()  # (H, W)
        heatmap = F.relu(heatmap)

        # Normalize
        heatmap = heatmap.cpu().detach()
        if heatmap.max() > 0:
            heatmap /= (heatmap.max() + 1e-8)

        # Resize to the original image size
        original_size = (img_tensor.shape[1], img_tensor.shape[2])  # (H, W)
        heatmap_resized = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0), size=original_size, mode='bilinear', align_corners=False
        ).squeeze().numpy()

        # Optional: save for debug
        # plt.imsave(f"debug_heatmap_{idx}_label{label}.png", heatmap_resized, cmap='jet')

        heatmaps_by_class[label].append(heatmap_resized)

    # Remove hooks after processing
    fwd_handle.remove()
    bwd_handle.remove()

    return heatmaps_by_class


def _execute_method_flow(
    model,
    heatmap_gen_set,
    attack_set,
    method_name: str,
    method_run_dir: Path,
    max_level: int,
    attack_patch_size: int,
    class_names: dict,
    dataset,
):
    """
    Executes the heatmap generation, attack, and analysis for a single specified method.
    """
    print(f"\n--- Starting Flow for Method: {method_name.upper()} ---")
    method_run_dir.mkdir(exist_ok=True)

    # 1. Generate heatmaps based on the specified method
    heatmaps_by_class = defaultdict(list)
    if method_name == 'gradcam':
        # Detect target layer automatically or manually
        target_layer = None
        if hasattr(ut, 'SimpleCNN') and isinstance(model, ut.SimpleCNN):
            target_layer = model.conv_layers[3]
        elif hasattr(ut, 'CIFAR_CNN') and isinstance(model, ut.CIFAR_CNN):
            target_layer = model.features[8]
        elif hasattr(ut, 'GTSRB_CNN') and isinstance(model, ut.GTSRB_CNN):
            target_layer = model.features[6]
        else:
            print(f"ERROR: Grad-CAM not configured for model {type(model).__name__}.")
            return

        print(f"Using Grad-CAM on layer: {target_layer}")
        heatmaps_by_class = _generate_gradcam_heatmaps_in_memory(model, heatmap_gen_set, target_layer)

    elif method_name == 'scaled_zz':
        for img_tensor, label, idx, orig_pred in tqdm(heatmap_gen_set, desc="Generating Heatmaps (Scaled ZZ)", unit="img"):
            img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
            records = []
            _recursive_scaled_zz(model, img_tensor.to(DEVICE), orig_pred, idx, label, max_level, records, 0, 0, img_w, img_h, 1)
            individual_heatmap = np.zeros((img_h, img_w), dtype=np.float32)
            for rec in records:
                if rec['changed']:
                    try:
                        parts = rec['zone_coords'].replace('[', '').split(']')
                        y0, y1 = map(int, parts[0].split('..'))
                        x0, x1 = map(int, parts[1].split('..'))
                        individual_heatmap[y0:y1+1, x0:x1+1] += 1
                    except:
                        continue
            heatmaps_by_class[label].append(individual_heatmap)
    else:
        print(f"ERROR: Unknown heatmap method '{method_name}'.")
        return

    # 2. Aggregate and Save heatmaps
    aggregated_heatmaps = {label: np.mean(heatmaps, axis=0) for label, heatmaps in heatmaps_by_class.items() if heatmaps}
    
    agg_plot_dir = method_run_dir / "aggregated_heatmaps"
    agg_plot_dir.mkdir(exist_ok=True)
    print(f"\n--- Saving Aggregated Heatmaps for {method_name.upper()} to: {agg_plot_dir} ---")
    for label, avg_heatmap in aggregated_heatmaps.items():
        norm_heatmap = avg_heatmap / (avg_heatmap.max() + 1e-8)
        plt.figure(figsize=(6, 6))
        plt.imshow(norm_heatmap, cmap='jet')
        plt.colorbar(label='Average Importance')
        plt.title(f"Aggregated Heatmap ({method_name.upper()}) - Class {class_names.get(label, label)}")
        plt.axis('off')
        plt.savefig(agg_plot_dir / f"aggregated_class_{label}.png", bbox_inches='tight')
        plt.close()

    # 3. Execute attack
    attack_results = []
    attack_plots_dir = method_run_dir / "attack_plots"
    attack_plots_dir.mkdir(exist_ok=True)

    for img_tensor, true_label, idx, _ in tqdm(attack_set, desc=f"Executing Attacks ({method_name.upper()})", unit="img"):
        if true_label not in aggregated_heatmaps:
            continue

        attack_heatmap = aggregated_heatmaps[true_label]
        pixels_changed, final_pred, perturbed_img = _run_single_pixel_attack(
            model, img_tensor.to(DEVICE), true_label, attack_heatmap, attack_patch_size
        )

        total_pixels = img_tensor.shape[1] * img_tensor.shape[2]
        attack_results.append({
            "image_id": idx, "true_label": class_names.get(true_label, true_label),
            "final_prediction": class_names.get(final_pred, final_pred),
            "pixels_changed": pixels_changed,
            "pixels_changed_percent": (pixels_changed / total_pixels) * 100,
        })

        # Plot and save attack visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        img_cpu, perturbed_cpu = img_tensor.cpu(), perturbed_img.cpu()
        axes[0].imshow(img_cpu.permute(1, 2, 0).squeeze(), cmap='gray')
        axes[0].set_title(f"Original (ID: {idx})\nPred: {class_names.get(true_label, true_label)}")
        axes[0].axis('off')
        axes[1].imshow((img_cpu - perturbed_cpu).abs().permute(1, 2, 0).squeeze(), cmap='hot')
        axes[1].set_title(f"{pixels_changed} Altered Pixels")
        axes[1].axis('off')
        axes[2].imshow(perturbed_cpu.permute(1, 2, 0).squeeze(), cmap='gray')
        axes[2].set_title(f"Attacked\nNew Pred: {class_names.get(final_pred, final_pred)}")
        axes[2].axis('off')
        fig.suptitle(f"Attack ({method_name.upper()}) - Class {true_label} -> {final_pred}", fontsize=16)
        plt.savefig(attack_plots_dir / f"attack_id_{idx}_class_{true_label}.png", bbox_inches='tight')
        plt.close(fig)

    # 4. Final report for the method
    if attack_results:
        df_results = pd.DataFrame(attack_results).sort_values(by="pixels_changed_percent")
        print(f"\n--- Final Attack Report for {method_name.upper()} ---")
        print(df_results.to_string(index=False))

        csv_path = method_run_dir / f"attack_summary_{method_name}.csv"
        df_results.to_csv(csv_path, index=False)

        avg_pixels_changed = df_results["pixels_changed_percent"].mean()
        print(f"\nSummary ({method_name.upper()}): On average, {avg_pixels_changed:.2f}% of pixels were altered.")

        analysis_class_names = {i: name for i, name in enumerate(dataset.classes)} if hasattr(dataset, 'classes') else None
        if analysis_class_names:
            _analyze_attack_report_plots(pd.read_csv(csv_path), method_run_dir / "attack_analysis_plots", class_names=analysis_class_names)
    
    print(f"\n✅ {method_name.upper()} flow completed. Results in: {method_run_dir}")


def run_integrated_heatmap_attack(
    model,
    dataset,
    run_dir: Path,
    num_images_per_class=20,
    max_level=4,
    attack_patch_size=1,
    class_names=None,
    split_ratio=0.85,
):
    """
    Runs the complete flow for BOTH Grad-CAM and Scaled ZZ.
    It uses the same data split for both methods to ensure a fair comparison.
    """
    model.eval()
    print(f"\n--- Starting Integrated Generation and Attack Flow for Grad-CAM and Scaled ZZ ---")

    # 1. Select and split correctly classified images (Done ONCE for both methods)
    if num_images_per_class < 2:
        num_images_per_class = 2

    all_selected_imgs = select_correctly_classified_images(model, dataset, num_images_per_class)
    if not all_selected_imgs:
        print("No CORRECT images selected. Aborting."); return

    images_by_class = defaultdict(list)
    for img_data in all_selected_imgs:
        images_by_class[img_data[1]].append(img_data)

    heatmap_gen_set, attack_set = [], []
    for label, images in images_by_class.items():
        random.shuffle(images)
        split_point = math.ceil(len(images) * split_ratio)
        heatmap_gen_set.extend(images[:split_point])
        attack_set.extend(images[split_point:])

    print(f"\nData split complete: {len(heatmap_gen_set)} images for heatmap generation, {len(attack_set)} for attack.")
    print("This data split will be used for BOTH methods.")
    
    if not attack_set:
        print("ERROR: The attack set is empty. Try increasing --num_images."); return

    # --- Common arguments for the method flow ---
    flow_args = {
        "model": model, "heatmap_gen_set": heatmap_gen_set, "attack_set": attack_set,
        "max_level": max_level, "attack_patch_size": attack_patch_size,
        "class_names": class_names, "dataset": dataset,
    }

    # 2. Execute flow for Grad-CAM
    _execute_method_flow(
        **flow_args,
        method_name='gradcam',
        method_run_dir=run_dir / "gradcam_flow"
    )

    # 3. Execute flow for Scaled ZZ
    _execute_method_flow(
        **flow_args,
        method_name='scaled_zz',
        method_run_dir=run_dir / "scaled_zz_flow"
    )

    print(f"\n\n✅✅✅ All flows completed. Main results directory: {run_dir}")