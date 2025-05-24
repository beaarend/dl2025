import torch
import torch.nn as nn
import torchvision.transforms as transforms
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
        # Não levanta erro para simplificar, mas a versão robusta poderia
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
    
    print(f"Iniciando seleção 1 por 1 (embaralhado): {num_per_class} imagens por classe...")

    # Tenta obter o número de classes do dataset
    try:
        num_target_classes = len(dataset.classes)
        all_possible_labels = list(range(num_target_classes))
        print(f"Dataset com {num_target_classes} classes alvo ({all_possible_labels}).")
    except AttributeError:
        print("AVISO: Não foi possível determinar o número exato de classes. Assumindo 10.")
        num_target_classes = 10 
        all_possible_labels = list(range(10))

    # Cria uma lista de índices e embaralha
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    processed_count = 0
    for idx in indices:
        # Pega a imagem e o label usando o índice embaralhado
        img_tensor, label = dataset[idx] 
        
        # Garante que o label seja um inteiro
        label_item = label if isinstance(label, int) else label.item()

        # Pula se já temos o suficiente para esta classe
        if counts[label_item] >= num_per_class:
            continue

        # Garante que é tensor (caso o dataset não retorne tensor)
        if not isinstance(img_tensor, torch.Tensor):
             img_tensor = transforms.ToTensor()(img_tensor).float()

        # Prepara a imagem para a predição
        input_for_pred = _preprocess_for_model(img_tensor.clone(), model)
        if input_for_pred is None: continue 

        # Realiza a predição
        with torch.no_grad():
            pred = model(input_for_pred.to(DEVICE).unsqueeze(0)).argmax(dim=1).item()

        processed_count += 1
        if processed_count % 500 == 0:
             print(f"  ... Verificadas {processed_count} imagens... Contagem: {dict(counts)}")

        # Adiciona se a predição for correta
        if pred == label_item:
            print(f"  -> Encontrada Idx {idx}: Classe {label_item} (Pred: {pred}) - OK")
            selected.append((img_tensor.cpu(), label_item, idx, pred)) 
            counts[label_item] += 1
        
        # Verifica se já coletamos o suficiente para todas as classes
        num_classes_completed = sum(1 for class_idx in all_possible_labels if counts[class_idx] >= num_per_class)
        
        if num_classes_completed >= num_target_classes:
            print("  -> Atingido o número desejado para todas as classes. Parando seleção.")
            break 
            
    if not selected:
         print("!!! ATENÇÃO: Nenhuma imagem selecionada.")
    else:
        print(f"Seleção 1 por 1 concluída. {len(selected)} imagens selecionadas.")
        for lbl in sorted(counts.keys()):
            print(f"  Classe {lbl}: {counts[lbl]} imagens selecionadas.")
             
    return selected

# --- Seleção de imagens (uma por classe) ---
def select_one_per_class(dataset, num_per_class=3):
    """
    Seleciona até `num_per_class` imagens por classe no dataset.
    Adaptado para funcionar com datasets que podem não ter 'target' diretamente
    (como ImageNet que usa tuplas (img, label)).

    Args:
        dataset (torch.utils.data.Dataset): Dataset para seleção.
        num_per_class (int): Quantidade máxima de imagens por classe.

    Returns:
        selected (list): Lista de tuplas (img, label, idx) selecionadas.
    """
    counts = defaultdict(int)
    selected = []

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        # Garante que a imagem é um tensor e tem 3 canais para modelos pré-treinados,
        # ou 1 para MNIST/FashionMNIST, e que está em float
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        elif not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img).float()

        if img.shape[0] == 1 and img.shape[0] != 3: # Se for grayscale e não RGB, duplicar canais
             img = img.repeat(3, 1, 1)

        # Redimensiona para 224x224 se a imagem for diferente e não for MNIST
        if img.shape[1] != 28 and img.shape[2] != 28:
            resize_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224)
            ])
            img = resize_transform(img)

        # Certifica-se de que a label é um int
        if isinstance(label, torch.Tensor):
            label = label.item()

        if counts[label] < num_per_class:
            selected.append((img, label, idx))
            counts[label] += 1
        # Se já pegou a quantidade desejada de todas as classes conhecidas, pode parar
        # Aqui, assumimos 10 classes para MNIST, mas para outros datasets, pode ser dinâmico
        # Para um caso genérico, removeria o 'len(counts) == 10'
        if len(counts) >= 10 and all(c >= num_per_class for c in counts.values()):
            break

    return selected


def recursive_zero_zones(model, img, orig_pred, img_id, label, run_dir,
                         x0, x1, y0, y1, level, max_level, records):
    H_zone_full = y1 - y0 
    W_zone_full = x1 - x0 
    
    orig_img_h, orig_img_w = img.shape[1], img.shape[2]

    if H_zone_full < 4 or W_zone_full < 4 or level > max_level:
        return

    zone_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    h_half = H_zone_full // 2
    w_half = W_zone_full // 2

    if h_half == 0 or w_half == 0: # Não pode dividir se uma das metades for zero
        return

    for zone_name_str in zone_names:
        if zone_name_str == 'top_left':
            sx0, sx1_excl = x0, x0 + w_half
            sy0, sy1_excl = y0, y0 + h_half
        elif zone_name_str == 'top_right':
            sx0, sx1_excl = x0 + w_half, x1
            sy0, sy1_excl = y0, y0 + h_half
        elif zone_name_str == 'bottom_left':
            sx0, sx1_excl = x0, x0 + w_half
            sy0, sy1_excl = y0 + h_half, y1
        else:  # bottom_right
            sx0, sx1_excl = x0 + w_half, x1
            sy0, sy1_excl = y0 + h_half, y1

        # Coordenadas para zerar (exclusivo no final)
        current_zone_h = sy1_excl - sy0
        current_zone_w = sx1_excl - sx0

        if current_zone_w <= 0 or current_zone_h <= 0:
            continue

        img_z = img.clone()
        img_z[:, sy0:sy1_excl, sx0:sx1_excl] = 0 
        
        # Coordenadas para display e log (inclusivo no final)
        zone_coords_display_str = f"[{sy0}..{sy1_excl-1}][{sx0}..{sx1_excl-1}]"

        input_tensor_for_pred = _preprocess_for_model(img_z, model)
        
        model.eval()
        with torch.no_grad():
            out = model(input_tensor_for_pred.to(DEVICE).unsqueeze(0))
        
        zone_pred_val = out.argmax(dim=1).item()
        changed_pred_flag = (zone_pred_val != orig_pred)
        
        records.append({
            'image_id': img_id, 
            'true_label': label, 
            'orig_pred': orig_pred,
            'zone_name': zone_name_str, 
            'zone_coords': zone_coords_display_str, 
            'zone_pred': zone_pred_val, 
            'changed': changed_pred_flag,
            'img_height': orig_img_h, 
            'img_width': orig_img_w,  
            'level': level,
            'zone_area_pixels': current_zone_h * current_zone_w
        })

        img_z_display = img_z.squeeze(0).cpu()
        img_z_display_clamped = torch.clamp(img_z_display, 0, 1)

        plt.figure(figsize=(7, 7)) 
        plt.imshow(img_z_display_clamped.permute(1,2,0) if img_z_display_clamped.shape[0] == 3 else img_z_display_clamped, 
                   cmap='gray' if img_z_display_clamped.shape[0] == 1 else None)
        
        title_zone = (f"ID:{img_id} (OrigRes:{orig_img_h}x{orig_img_w}) Lvl:{level} Zone:{zone_name_str}\n"
                      f"GT:{label} OrigP:{orig_pred} ZoneP:{zone_pred_val} Changed:{changed_pred_flag}\n"
                      f"Coords: {zone_coords_display_str}")
        plt.title(title_zone, fontsize=8) 
        plt.axis('off')
        
        image_save_dir = run_dir / "zero_zones_images" / f"img_id_{img_id}" 
        image_save_dir.mkdir(parents=True, exist_ok=True)
        
        filename_zone = f"zz_id{img_id}_lvl{level}_{zone_name_str}_origP{orig_pred}_zoneP{zone_pred_val}.png"
        plt.savefig(image_save_dir / filename_zone, bbox_inches='tight')
        plt.close()

        if changed_pred_flag:
            recursive_zero_zones(model, img, orig_pred, img_id, label, run_dir,
                                 sx0, sx1_excl, sy0, sy1_excl, level+1, max_level, records)

def parse_coords(coords_str):
    try:
        # Remove colchetes e divide pelas partes de linha e coluna
        parts = coords_str.replace('[', '').split(']')
        rows_part = parts[0]
        cols_part = parts[1]
        
        r0, r1 = map(int, rows_part.split('..'))
        c0, c1 = map(int, cols_part.split('..'))
        return r0, r1, c0, c1
    except Exception as e: # Captura exceções mais genéricas durante o parse
        print(f"AVISO: Falha ao parsear coordenadas: '{coords_str}'. Erro: {e}. Retornando 0s.")
        return 0,0,0,0

def generate_heatmap_overlay(records, img_tensor, img_id, run_dir, orig_pred_for_title):
    orig_img_h, orig_img_w = img_tensor.shape[1], img_tensor.shape[2]
    heatmap = np.zeros((orig_img_h, orig_img_w), dtype=float)
    
    changed_zones_count = 0
    current_image_records = [rec for rec in records if rec['image_id'] == img_id]

    if not current_image_records: # Se não há records para esta imagem, não faz nada
        print(f"Aviso: Sem records para heatmap da imagem ID {img_id}.")
        return

    true_label_for_title = current_image_records[0]['true_label'] # Pega de qualquer record da imagem

    for rec in current_image_records:
        if rec['changed']: 
            r0, r1, c0, c1 = parse_coords(rec['zone_coords'])
             # Adiciona 1 a r1 e c1 porque parse_coords retorna inclusivo, e slice é exclusivo no final
            heatmap[r0:min(r1 + 1, orig_img_h), c0:min(c1 + 1, orig_img_w)] += 1
            changed_zones_count +=1
    
    if heatmap.max() > 0:
        heatmap_norm = heatmap / heatmap.max()
    else:
        heatmap_norm = heatmap 

    img_display_overlay = img_tensor.squeeze(0).cpu()
    img_display_overlay_clamped = torch.clamp(img_display_overlay, 0, 1)
    img_np = img_display_overlay_clamped.permute(1,2,0).numpy() if img_display_overlay_clamped.shape[0]==3 else img_display_overlay_clamped.numpy()

    plt.figure(figsize=(8,8)) 
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

# --- Funções de Zero Zones ---
def generate_zero_zone_analysis(model, dataset, run_dir: Path, num_images_per_class=1, max_level=1000):
    model.eval() 
    xai_records = []
    
    selected_imgs = select_correctly_classified_images(model, dataset, num_images_per_class)    

    if not selected_imgs:
        print("Nenhuma imagem CORRETA selecionada para análise de Zero Zones.")
        return
    
    print(f"Iniciando análise Zero Zones para {len(selected_imgs)} imagens CORRETAS...")

    for img_tensor, label, idx, orig_pred in selected_imgs:
        img_tensor = img_tensor.to(DEVICE)
        
        img_h, img_w = img_tensor.shape[1], img_tensor.shape[2] 

        img_display_orig = img_tensor.squeeze(0).cpu()
        img_display_orig_clamped = torch.clamp(img_display_orig, 0, 1) 

        plt.figure(figsize=(7, 7)) 
        plt.imshow(img_display_orig_clamped.permute(1,2,0) if img_display_orig_clamped.shape[0] == 3 else img_display_orig_clamped, 
                   cmap='gray' if img_display_orig_clamped.shape[0] == 1 else None)
        
        title_orig = f"ID:{idx} - Orig Img ({img_h}x{img_w})\nGT:{label} Pred:{orig_pred}"
        plt.title(title_orig, fontsize=10)
        plt.axis('off')
        
        base_images_dir = run_dir / "base_images"
        base_images_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(base_images_dir / f"orig_id{idx}_label{label}_pred{orig_pred}_res{img_h}x{img_w}.png", bbox_inches='tight')
        plt.close()

        current_image_records = []
        recursive_zero_zones(model, img_tensor, orig_pred, idx, label, run_dir,
                             0, img_w, 0, img_h, level=1, max_level=max_level, records=current_image_records)

        xai_records.extend(current_image_records) 

        generate_heatmap_overlay(current_image_records, img_tensor, idx, run_dir, orig_pred) 

    if xai_records: 
        df_base = pd.DataFrame([{
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analyzed_image_ids': sorted(list(set([r['image_id'] for r in xai_records]))),
            'num_analyzed_images': len(set([r['image_id'] for r in xai_records]))
        }])
        df_base.to_csv(run_dir / "summary.csv", index=False)

        df_xai = pd.DataFrame(xai_records)
        if not df_xai.empty: # Garante que o DataFrame não está vazio antes de adicionar coluna
            df_xai['orig_img_resolution'] = df_xai.apply(lambda row: f"{row['img_height']}x{row['img_width']}", axis=1)
        df_xai = df_xai.sort_values(by=['image_id', 'level', 'zone_name'], ascending=[True, True, True]) 
        df_xai.to_csv(run_dir / "zero_zones_xai.csv", index=False)
        
    print(f"Análise Zero Zones concluída. Resultados salvos em: {run_dir}")