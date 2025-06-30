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
import random # <--- Importar random
from tqdm import tqdm # <--- tqdm já estava importado

try:
    import utils as ut # Tenta importar utils para get_model_input_channels
    HAS_UTILS = True
except ImportError:
    print("AVISO: Não foi possível importar 'utils.py'. A lógica de canais pode ser limitada.")
    HAS_UTILS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _preprocess_for_model(img_tensor, model):
    """
    Função auxiliar para pré-processar UMA imagem para predição.
    """
    input_tensor = img_tensor.clone()
    expected_channels = -1 

    try:
        if HAS_UTILS:
            expected_channels = ut.get_model_input_channels(model)
        elif hasattr(model, 'conv1'): 
            expected_channels = model.conv1.in_channels
        else: 
             model_type_str = str(type(model)).lower()
             if 'resnet' in model_type_str or 'mobile' in model_type_str: expected_channels = 3
             else: expected_channels = 1
            
        current_channels = input_tensor.shape[0]

        if current_channels == 1 and expected_channels == 3:
            input_tensor = input_tensor.repeat(3, 1, 1)
        elif current_channels == 3 and expected_channels == 1:
            input_tensor = input_tensor.mean(dim=0, keepdim=True)
    except Exception as e:
        print(f"AVISO Preprocess (simplificado): Falha ({e}).")
        if 'SimpleCNN' not in str(type(model)) and input_tensor.shape[0] == 1:
            input_tensor = input_tensor.repeat(3, 1, 1)
        expected_channels = input_tensor.shape[0] 

    if expected_channels == 3 and (input_tensor.shape[1:] != (224, 224)):
         resize_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
         input_tensor = resize_transform(input_tensor)
    elif expected_channels == 1 and 'SimpleCNN' in str(type(model)) and (input_tensor.shape[1:] != (28, 28)):
         if input_tensor.shape[1:] != (28,28):
             print(f"AVISO: SimpleCNN com input de tamanho {input_tensor.shape[1:]}, esperado 28x28.")
         
    return input_tensor

def select_correctly_classified_images(model, dataset, num_per_class=1):
    """
    Seleciona até `num_per_class` imagens CORRETAMENTE CLASSIFICADAS
    por classe, processando 1 por 1 (embaralhado) e parando cedo.
    """
    model.eval()
    counts = defaultdict(int)
    selected = []
    
    try:
        num_target_classes = len(dataset.classes)
        all_possible_labels = list(range(num_target_classes))
    except AttributeError:
        print("AVISO: Não foi possível determinar o número exato de classes. Assumindo 10.")
        num_target_classes = 10 
        all_possible_labels = list(range(10))

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # --- MUDANÇA AQUI: Adicionando tqdm ---
    progress_bar = tqdm(indices, desc=f"Buscando {num_per_class} img/classe", unit="img", dynamic_ncols=True)

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
            # Atualiza informações na barra de progresso
            progress_bar.set_postfix(found=len(selected), classes=f"{len(counts)}/{num_target_classes}")
        
        num_classes_completed = sum(1 for class_idx in all_possible_labels if counts[class_idx] >= num_per_class)
        
        if num_classes_completed >= num_target_classes:
            progress_bar.close()
            print(f"\n  -> Atingido o número desejado para todas as classes ({len(selected)} imagens). Parando seleção.")
            break 
            
    if not selected:
         print("!!! ATENÇÃO: Nenhuma imagem selecionada.")
    else:
        print(f"Seleção concluída. {len(selected)} imagens selecionadas.")
        for lbl in sorted(counts.keys()):
            print(f"  Classe {lbl}: {counts[lbl]} imagens selecionadas.")
             
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
        print(f"AVISO: Falha ao parsear coordenadas: '{coords_str}'. Erro: {e}. Retornando 0s.")
        return 0, 0, 0, 0

def generate_heatmap_overlay(records, img_tensor, img_id, run_dir, orig_pred_for_title, show_matrix=False):
    orig_img_h, orig_img_w = img_tensor.shape[1], img_tensor.shape[2]
    heatmap = np.zeros((orig_img_h, orig_img_w), dtype=float)
    changed_zones_count = 0
    current_image_records = [rec for rec in records if rec['image_id'] == img_id]
    if not current_image_records:
        # print(f"Aviso: Sem records para heatmap da imagem ID {img_id}.")
        return
    true_label_for_title = current_image_records[0]['true_label']
    for rec in current_image_records:
        if rec['changed']:
            r0, r1, c0, c1 = parse_coords(rec['zone_coords'])
            heatmap[r0:min(r1 + 1, orig_img_h), c0:min(c1 + 1, orig_img_w)] += 1
            changed_zones_count += 1
    
    if show_matrix:
        print(f"\n--- Matriz Numérica do Heatmap 'Zero Zones' (ID: {img_id}) ---")
        print("Valores representam a contagem de vezes que uma zona recursiva causou mudança na predição.")
        # Para matrizes pequenas (como MNIST 28x28), a impressão é legível.
        # Para maiores, pode ser truncada pelo NumPy.
        np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=0)
        print(heatmap)
        np.set_printoptions() 
        print("--- Fim da Matriz ---\n")            

    if heatmap.max() > 0: heatmap_norm = heatmap / heatmap.max()
    else: heatmap_norm = heatmap
    img_display_overlay = img_tensor.squeeze(0).cpu()
    img_display_overlay_clamped = torch.clamp(img_display_overlay, 0, 1)
    img_np = img_display_overlay_clamped.permute(1, 2, 0).numpy() if img_display_overlay_clamped.shape[0] == 3 else img_display_overlay_clamped.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np, cmap='gray' if img_tensor.shape[0] == 1 else None, interpolation='nearest')
    plt.imshow(heatmap_norm, cmap='jet', interpolation='nearest', alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(label='Frequência de Mudança de Predição (Normalizada)')
    title_heatmap = (f"Heatmap - ID:{img_id} (OrigRes:{orig_img_h}x{orig_img_w})\n"
                     f"TrueLabel:{true_label_for_title} OrigPred:{orig_pred_for_title} "
                     f"({changed_zones_count} zonas mudaram predição)")
    plt.title(title_heatmap, fontsize=10)
    plt.axis('off')
    heatmaps_dir = run_dir / "heatmaps_overlay"
    heatmaps_dir.mkdir(exist_ok=True)
    plt.savefig(heatmaps_dir / f"overlay_heatmap_id{img_id}_res{orig_img_h}x{orig_img_w}.png", bbox_inches='tight')
    plt.close()

def _compute_zero_zone_matrix(model, img_tensor, orig_pred, idx, label, max_level):
    """Refatorado: Calcula e retorna a matriz do heatmap para Zero Zones de UMA imagem."""
    img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
    records = []
    # A função recursiva preenche a lista 'records'
    # Passamos um run_dir dummy pois não queremos salvar as imagens individuais da recursão aqui.
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
        print("Nenhuma imagem CORRETA selecionada para análise de Zero Zones.")
        return

    print(f"\nIniciando análise Zero Zones para {len(selected_imgs)} imagens...")
    
    # Dicionário para agrupar heatmaps por classe antes de salvar
    heatmaps_by_class = defaultdict(list)

    for img_tensor, label, idx, orig_pred in tqdm(selected_imgs, desc="Gerando Heatmaps Zero-Zone", unit="img"):
        # Gera a matriz do heatmap individual SEM salvar plots intermediários da recursão
        individual_heatmap = _compute_zero_zone_matrix(model, img_tensor, orig_pred, idx, label, max_level)
        
        # Adiciona o heatmap à lista de sua classe
        heatmaps_by_class[label].append(individual_heatmap)

        # A plotagem do heatmap individual (opcional) ainda pode ser feita aqui se desejado
        # generate_heatmap_overlay(...)

    # --- LÓGICA DE SALVAMENTO .NPY ---
    npy_output_dir = run_dir / "individual_heatmaps_npy"
    npy_output_dir.mkdir(exist_ok=True)
    print(f"\nSalvando heatmaps individuais em arrays .npy em: {npy_output_dir}")

    for label, heatmap_list in heatmaps_by_class.items():
        # Empilha a lista de matrizes 2D em uma única matriz 3D
        stacked_heatmaps = np.stack(heatmap_list, axis=0)
        
        save_path = npy_output_dir / f"heatmaps_class_{label}.npy"
        np.save(save_path, stacked_heatmaps)
        print(f"  -> Classe {label}: Salvo array com shape {stacked_heatmaps.shape} em {save_path.name}")
    
    print("\nAnálise Zero Zones (geração de .npy) concluída.")

#precisa atualizar...
def generate_knn_heatmap(model, dataset, run_dir: Path, num_images_per_class: int = 1, k_neighbors_to_paint: int = 10, num_key_pixels_to_evaluate: int = 200, perturb_patch_size: int = 5, show_matrix: bool = False):
    model.eval()
    selected_imgs_data = select_correctly_classified_images(model, dataset, num_images_per_class)
    if not selected_imgs_data:
        print("Nenhuma imagem CORRETAMENTE CLASSIFICADA selecionada para análise de heatmap KNN.")
        return
    knn_heatmaps_dir = run_dir / "knn_heatmaps_generated"
    knn_heatmaps_dir.mkdir(parents=True, exist_ok=True)
    base_images_dir = run_dir / "base_images_for_knn"
    base_images_dir.mkdir(parents=True, exist_ok=True)
    summary_records = []
    print(f"\nIniciando geração de KNN Heatmaps (k={k_neighbors_to_paint}, pixels_teste={num_key_pixels_to_evaluate})")

    # --- MUDANÇA AQUI: tqdm no loop principal de imagens ---
    for img_tensor, true_label, original_idx, orig_pred_class in tqdm(selected_imgs_data, desc="Gerando KNN Heatmaps", unit="img"):
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

        # --- MUDANÇA AQUI: tqdm no loop interno de pixels (o mais demorado) ---
        # leave=False faz a barra interna sumir ao terminar, limpando a tela.
        pixel_iterator = tqdm(key_pixels_to_test, desc=f"  Analisando pixels ID {original_idx}", leave=False, unit="pixel", dynamic_ncols=True)
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
            print(f"\n--- Matriz Numérica do Heatmap 'KNN' (ID: {original_idx}) ---")
            print("Valores representam a contagem de 'votos' de pixels vizinhos a um ponto de impacto.")
            np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=0)
            print(current_image_heatmap)
            np.set_printoptions() 
            print("--- Fim da Matriz ---\n")
        
        summary_records.append({'image_id': original_idx, 'true_label': true_label, 'original_prediction': orig_pred_class, 'num_key_pixels_evaluated': len(key_pixels_to_test), 'num_impactful_key_pixels': impactful_pixels_found_count, 'prediction_changed_by_perturbation': impactful_pixels_found_count > 0})
        
        if current_image_heatmap.max() > 0: heatmap_normalized = current_image_heatmap / current_image_heatmap.max()
        else: heatmap_normalized = current_image_heatmap

        display_original_img_np = img_tensor.cpu().permute(1, 2, 0).numpy() if img_tensor.shape[0] == 3 else img_tensor.cpu().squeeze(0).numpy()
        display_original_img_np = np.clip(display_original_img_np, 0, 1)

        plt.figure(figsize=(8, 8))
        plt.imshow(display_original_img_np, cmap='gray' if img_tensor.shape[0] == 1 else None, interpolation='nearest')
        plt.imshow(heatmap_normalized, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        plt.colorbar(label=f'Impacto KNN (k={k_neighbors_to_paint})')
        title_str = (f"KNN Heatmap - ID:{original_idx}\n"
                     f"TrueLabel:{true_label}, OrigPred:{orig_pred_class}\n"
                     f"Perturb Patch: {perturb_patch_size}x{perturb_patch_size}, Pixels Avaliados: {len(key_pixels_to_test)}")
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
            print(f"Sumário da análise KNN Heatmap salvo em: {summary_csv_path}")
        except Exception as e:
            print(f"Erro ao salvar o sumário CSV da análise KNN: {e}")
    else:
        print("Nenhum dado de sumário para salvar para a análise KNN.")
    print(f"Análise de Heatmap KNN concluída. Resultados em: {run_dir}")
    
    
def aggregate_and_plot_from_npy(npy_input_dir: Path, output_dir: Path):
    """
    Lê arquivos .npy contendo heatmaps, agrega-os por classe, e salva os plots finais.
    Esta é uma função independente para o Passo 2.
    """
    print(f"\n--- Iniciando Passo 2: Agregação a partir de arquivos .npy ---")
    print(f"Lendo arquivos de: {npy_input_dir}")
    print(f"Salvando plots agregados em: {output_dir}")
    
    output_dir.mkdir(exist_ok=True)
    
    npy_files = list(npy_input_dir.glob("heatmaps_class_*.npy"))
    if not npy_files:
        print(f"AVISO: Nenhum arquivo 'heatmaps_class_*.npy' encontrado em {npy_input_dir}")
        return

    for file_path in tqdm(npy_files, desc="Processando classes", unit="file"):
        try:
            # Extrai o label da classe do nome do arquivo
            label = int(file_path.stem.split('_')[-1])
        except (ValueError, IndexError):
            print(f"AVISO: Não foi possível extrair o label do arquivo {file_path.name}. Pulando.")
            continue

        # Carrega o array de heatmaps
        stacked_heatmaps = np.load(file_path)
        num_aggregated = stacked_heatmaps.shape[0]

        # Calcula o heatmap médio
        average_heatmap = np.mean(stacked_heatmaps, axis=0)

        # Normaliza para visualização
        if average_heatmap.max() > 0:
            viz_heatmap = average_heatmap / average_heatmap.max()
        else:
            viz_heatmap = average_heatmap

        print(f"Classe {label}: {num_aggregated} heatmaps agregados. Valor máx. da média: {average_heatmap.max():.2f}")

        # Plota e salva o resultado final
        plt.figure(figsize=(6, 6))
        plt.imshow(viz_heatmap, cmap='jet', interpolation='nearest')
        plt.colorbar(label='Importância Média Normalizada')
        plt.title(f"Heatmap Agregado - Classe {label}\n({num_aggregated} imagens)")
        plt.axis('off')
        
        img_path = output_dir / f"aggregated_class_{label}.png"
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

    print(f"\nAgregação e plotagem concluídas. Resultados em {output_dir}.")   
    
# --- NOVA FUNCIONALIDADE: Ataque de Pixel Guiado por Heatmap ---

def _run_single_pixel_attack(model, image_tensor, true_label, attack_heatmap):
    """
    Executa o ataque de perturbação em uma única imagem, zerando pixels
    em ordem de importância do heatmap até que a predição mude.

    Retorna:
        - pixels_changed (int): Número de pixels modificados.
        - final_prediction (int): A nova predição (incorreta).
        - perturbed_image (Tensor): A imagem com os pixels zerados.
    """
    # 1. Obter a ordem dos pixels a serem atacados
    # Flatten o heatmap e obtenha os índices que o ordenariam em ordem decrescente
    flat_heatmap = attack_heatmap.flatten()
    # argsort ordena do menor para o maior, então invertemos com [::-1]
    sorted_pixel_indices = np.argsort(flat_heatmap)[::-1]
    
    # Prepara a imagem para a perturbação
    perturbed_image = image_tensor.clone()
    num_total_pixels = image_tensor.shape[1] * image_tensor.shape[2]

    # 2. Iterar e atacar pixel por pixel
    for i, flat_idx in enumerate(sorted_pixel_indices):
        pixels_changed = i + 1
        
        # Converte o índice flat de volta para coordenadas (linha, coluna)
        row, col = np.unravel_index(flat_idx, attack_heatmap.shape)
        
        # Zera o pixel na imagem (em todos os canais)
        perturbed_image[:, row, col] = 0.0 # Zerar o pixel
        
        # 3. Reavalia o modelo com a imagem perturbada
        with torch.no_grad():
            input_for_pred = _preprocess_for_model(perturbed_image.clone(), model)
            new_pred = model(input_for_pred.to(DEVICE).unsqueeze(0)).argmax().item()
            
        # 4. Verifica se o ataque foi bem-sucedido
        if new_pred != true_label:
            return pixels_changed, new_pred, perturbed_image
            
    # Se o loop terminar, significa que mesmo zerando todos os pixels, a predição não mudou.
    return num_total_pixels, true_label, perturbed_image


def generate_pixel_attack_report(model, dataset, aggregated_heatmaps_dir: Path, run_dir: Path, num_images_per_class: int):
    """
    Orquestra o teste de ataque de pixel para várias imagens e gera um relatório.
    """
    print("--- Iniciando Análise de Ataque de Pixel Guiado por Heatmap ---")
    
    # 1. Carregar os heatmaps agregados
    aggregated_heatmaps = {}
    npy_files = list(aggregated_heatmaps_dir.glob("*.npy"))
    if not npy_files:
        print(f"ERRO: Nenhum arquivo de heatmap .npy encontrado em '{aggregated_heatmaps_dir}'.")
        print("Execute primeiro a análise 'aggregate_only'.")
        return
        
    for file_path in npy_files:
        try:
            label = int(file_path.stem.split('_')[-1])
            aggregated_heatmaps[label] = np.load(file_path)
        except:
            continue
    print(f"Carregados {len(aggregated_heatmaps)} heatmaps agregados de {aggregated_heatmaps_dir}.")

    # 2. Selecionar imagens de teste (que o modelo acerta)
    test_images = select_correctly_classified_images(model, dataset, num_images_per_class)
    if not test_images:
        print("Nenhuma imagem corretamente classificada encontrada para atacar.")
        return

    attack_results = []
    output_plots_dir = run_dir / "pixel_attack_plots"
    output_plots_dir.mkdir(exist_ok=True)

    # 3. Executar o ataque para cada imagem de teste
    for img_tensor, true_label, idx, _ in tqdm(test_images, desc="Executando Ataques"):
        if true_label not in aggregated_heatmaps:
            continue

        # Obtém o heatmap agregado para a classe desta imagem
        # Usamos a média dos heatmaps para o ataque
        attack_heatmap = np.mean(aggregated_heatmaps[true_label], axis=0)

        # Roda o ataque
        pixels_changed, final_pred, perturbed_img = _run_single_pixel_attack(model, img_tensor, true_label, attack_heatmap)
        
        total_pixels = img_tensor.shape[1] * img_tensor.shape[2]
        attack_results.append({
            "image_id": idx,
            "true_label": true_label,
            "final_prediction": final_pred,
            "pixels_changed": pixels_changed,
            "pixels_changed_percent": (pixels_changed / total_pixels) * 100
        })

        # 4. Gerar um plot de resultado visual para este ataque
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Imagem Original
        axes[0].imshow(img_tensor.cpu().permute(1, 2, 0).squeeze(), cmap='gray')
        axes[0].set_title(f"Original (ID: {idx})\nPred: {true_label}")
        axes[0].axis('off')
        
        # Máscara de Perturbação
        mask = (img_tensor - perturbed_img).abs().cpu().permute(1, 2, 0).squeeze()
        axes[1].imshow(mask, cmap='hot')
        axes[1].set_title(f"{pixels_changed} Pixels Alterados")
        axes[1].axis('off')

        # Imagem Atacada
        axes[2].imshow(perturbed_img.cpu().permute(1, 2, 0).squeeze(), cmap='gray')
        axes[2].set_title(f"Atacada\nNova Pred: {final_pred}")
        axes[2].axis('off')

        fig.suptitle(f"Resultado do Ataque - Classe {true_label} -> {final_pred}", fontsize=16)
        plt.savefig(output_plots_dir / f"attack_id_{idx}_class_{true_label}.png", bbox_inches='tight')
        plt.close(fig)

    # 5. Apresentar um relatório final no console
    if attack_results:
        df_results = pd.DataFrame(attack_results)
        df_results = df_results.sort_values(by="pixels_changed_percent")
        
        print("\n--- Relatório Final do Ataque de Pixel ---")
        print(df_results.to_string(index=False))
        
        # Salva o relatório em CSV
        df_results.to_csv(run_dir / "pixel_attack_summary.csv", index=False)
        
        avg_pixels_changed = df_results["pixels_changed_percent"].mean()
        print(f"\nResumo: Em média, foi necessário alterar {avg_pixels_changed:.2f}% dos pixels para enganar o modelo.")
    
    print(f"Plots dos ataques salvos em: {output_plots_dir}")     