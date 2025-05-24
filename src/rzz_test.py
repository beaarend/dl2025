import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict
import math

# --- Configurações gerais ---
BASE_RESULTS_DIR = Path("results")
BASE_RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helpers de diretório ---
def get_next_run_dir(base_dir=BASE_RESULTS_DIR):
    runs = [d for d in base_dir.iterdir() if d.name.startswith("run") and d.is_dir()]
    ids = [int(d.name.replace("run", "")) for d in runs if d.name.replace("run", "").isdigit()]
    run_id = max(ids, default=0) + 1
    run_dir = base_dir / f"run{run_id}"
    run_dir.mkdir()
    return run_dir

# --- Modelo ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)

        # calcula flatten dim automaticamente
        dummy = torch.zeros(1, 1, 28, 28)
        x = self.conv1(dummy)
        x = self.conv2(x)
        flat_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --- Dados ---
TRANSFORM = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=TRANSFORM)
test_set  = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=TRANSFORM)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)

# --- Treino e avaliação de época ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total


def eval_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct/total

# --- Seleção de imagens (uma por classe) ---
def select_one_per_class(dataset, num_per_class=3):
    """
    Seleciona até `num_per_class` imagens por classe no dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset para seleção.
        num_per_class (int): Quantidade máxima de imagens por classe.

    Returns:
        selected (list): Lista de tuplas (img, label, idx) selecionadas.
    """
    from collections import defaultdict

    counts = defaultdict(int)
    selected = []

    for idx, (img, label) in enumerate(dataset):
        if counts[label] < num_per_class:
            selected.append((img, label, idx))
            counts[label] += 1
        # Se já pegou a quantidade desejada de todas as 10 classes, pode parar
        if len(counts) == 10 and all(c >= num_per_class for c in counts.values()):
            break

    return selected

def select_all(test_set):
    """
    Retorna uma lista com todas as imagens, labels e seus índices do dataset.

    Args:
        test_set (torch.utils.data.Dataset): Dataset de teste.

    Returns:
        selected (list of tuples): Lista com tuplas (imagem, label, índice).
    """
    selected = []
    for idx in range(len(test_set)):
        img, label = test_set[idx]
        selected.append((img, label, idx))
    return selected

# --- Heurística Zero Zones Nível 1 ---
ZONE_NAMES = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

def apply_zero_zone(img, zone):
    img_z = img.clone()
    h, w = img.shape[1], img.shape[2]
    h_half, w_half = h // 2, w // 2
    if zone == 'top_left':
        coords = f"[0..{h_half-1}][0..{w_half-1}]"
        img_z[:, :h_half, :w_half] = 0
    elif zone == 'top_right':
        coords = f"[0..{h_half-1}][{w_half}..{w-1}]"
        img_z[:, :h_half, w_half:] = 0
    elif zone == 'bottom_left':
        coords = f"[{h_half}..{h-1}][0..{w_half-1}]"
        img_z[:, h_half:, :w_half] = 0
    else:
        coords = f"[{h_half}..{h-1}][{w_half}..{w-1}]"
        img_z[:, h_half:, w_half:] = 0
    return img_z, coords

# --- Funções de Subzonas ---
def apply_zero_zone_subblock(img, x0, x1, y0, y1, subzone):
    img_z = img.clone()
    h_block = y1 - y0
    w_block = x1 - x0
    h_half = h_block // 2
    w_half = w_block // 2
    if subzone == 'top_left':
        coords = f"[{y0}..{y0+h_half-1}][{x0}..{x0+w_half-1}]"
        img_z[:, y0:y0+h_half, x0:x0+w_half] = 0
    elif subzone == 'top_right':
        coords = f"[{y0}..{y0+h_half-1}][{x0+w_half}..{x1-1}]"
        img_z[:, y0:y0+h_half, x0+w_half:x1] = 0
    elif subzone == 'bottom_left':
        coords = f"[{y0+h_half}..{y1-1}][{x0}..{x0+w_half-1}]"
        img_z[:, y0+h_half:y1, x0:x0+w_half] = 0
    else:
        coords = f"[{y0+h_half}..{y1-1}][{x0+w_half}..{x1-1}]"
        img_z[:, y0+h_half:y1, x0+w_half:x1] = 0
    return img_z, coords

# --- Mapeamento Zona para Retângulo ---
def zone_to_rect(img, zone_name):
    H, W = img.shape[1], img.shape[2]
    midH, midW = H//2, W//2
    if zone_name == 'top_left':
        return (0, midW, 0, midH)
    elif zone_name == 'top_right':
        return (midW, W, 0, midH)
    elif zone_name == 'bottom_left':
        return (0, midW, midH, H)
    else:
        return (midW, W, midH, H)

def recursive_zero_zones(model, img, orig_pred, img_id, label, run_dir,
                         x0, x1, y0, y1, level, max_level, records):
    H = y1 - y0
    W = x1 - x0
    if H <= 1 or W <= 1 or level > max_level:
        return

    zones = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    h_half = H // 2
    w_half = W // 2

    for zone in zones:
        # calcula coords da subzona
        if zone == 'top_left':
            sx0, sx1 = x0, x0 + w_half
            sy0, sy1 = y0, y0 + h_half
        elif zone == 'top_right':
            sx0, sx1 = x0 + w_half, x1
            sy0, sy1 = y0, y0 + h_half
        elif zone == 'bottom_left':
            sx0, sx1 = x0, x0 + w_half
            sy0, sy1 = y0 + h_half, y1
        else:  # bottom_right
            sx0, sx1 = x0 + w_half, x1
            sy0, sy1 = y0 + h_half, y1

        # aplica zero zone na subzona atual
        img_z = img.clone()
        img_z[:, sy0:sy1, sx0:sx1] = 0
        coords = f"[{sy0}..{sy1-1}][{sx0}..{sx1-1}]"

        pred = model(img_z.unsqueeze(0).to(DEVICE)).argmax(dim=1).item()
        changed = (pred != orig_pred)
        records.append({
            'image_id': img_id, 'true': label, 'orig_pred': orig_pred,
            'zone': zone, 'zone_pred': pred, 'changed': changed,
            'img_height': img.shape[1], 'img_width': img.shape[2], 'coords': coords,
            'level': level
        })

        # Salva imagem
        plt.imshow(img_z.squeeze(0).cpu(), cmap='gray')
        plt.title(f"GT:{label} P:{pred} {coords} Lvl:{level}")
        plt.axis('off')
        (run_dir / "zero_zones_images").mkdir(parents=True, exist_ok=True)
        plt.savefig(run_dir / "zero_zones_images" / f"zz_lvl{level}_id{img_id}_{zone}_pred{pred}.png")
        plt.close()

        # Se mudou, chama recursivamente para essa subzona
        if changed:
            recursive_zero_zones(model, img, orig_pred, img_id, label, run_dir,
                                 sx0, sx1, sy0, sy1, level+1, max_level, records)

def parse_coords(coords):
    # Exemplo: "[0..13][0..13]"
    rows_part, cols_part = coords.split('][')
    rows_part = rows_part.strip('[')
    cols_part = cols_part.strip(']')
    r0, r1 = map(int, rows_part.split('..'))
    c0, c1 = map(int, cols_part.split('..'))
    return r0, r1, c0, c1

def generate_heatmap(records, img_h=28, img_w=28):
    heatmap = np.zeros((img_h, img_w), dtype=float)
    for rec in records:
        if rec['changed']:
            r0, r1, c0, c1 = parse_coords(rec['coords'])
            heatmap[r0:r1+1, c0:c1+1] += 1
    return heatmap

def get_max_levels(height, width):
    return int(math.floor(math.log2(min(height, width))))

def generate_zero_zone_analysis(model, dataset, run_dir, max_level=1000):
    model.eval()
    records = []

    (run_dir / "zero_zones_images").mkdir(exist_ok=True)

    selected_imgs = select_one_per_class(dataset)
    for img, label, idx in selected_imgs:
        img = img.to(DEVICE)
        orig_pred = model(img.unsqueeze(0)).argmax(dim=1).item()

        # salva imagem original
        plt.imshow(img.squeeze(0).cpu(), cmap='gray')
        plt.title(f"GT:{label} Pred:{orig_pred}")
        plt.axis('off')
        plt.savefig(run_dir / f"img{idx}_label{label}_pred{orig_pred}.png")
        plt.close()

        # executa análise recursiva
        recursive_zero_zones(
            model=model,
            img=img,
            orig_pred=orig_pred,
            img_id=idx,
            label=label,
            run_dir=run_dir,
            x0=0, x1=img.shape[2], y0=0, y1=img.shape[1],
            level=1,
            max_level=max_level,
            records=records
        )

    # salva CSV com estatísticas
    df = pd.DataFrame(records)
    df.to_csv(run_dir / "zero_zones_summary.csv", index=False)
    print(f"Análise de {len(selected_imgs)} imagens salva em: {run_dir}")

# --- Pipeline principal ---
def main():
    model = SimpleCNN().to(DEVICE)
    path = 'mnist_cnn.pth'
    if not os.path.exists(path):
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        for ep in range(1, 4):
            loss, acc_tr = train_one_epoch(model, train_loader, opt, crit)
            acc_te = eval_model(model, test_loader)
            print(f"[Ep {ep}] Loss: {loss:.4f}, Train Acc: {acc_tr:.4f}, Test Acc: {acc_te:.4f}")
        torch.save(model.state_dict(), path)
    else:
        model.load_state_dict(torch.load(path, map_location=DEVICE))

    final_acc = eval_model(model, test_loader)
    print(f"Final Test Accuracy: {final_acc:.4f}")
    run_dir = get_next_run_dir()

    selected = select_one_per_class(test_set, 10)
    xai_records = []

    max_levels = 10000  # ou o máximo que quiser
    # Zerar a imagem toda inicialmente (x0,y0,x1,y1) = (0,0, width, height)
    for img, label, idx in selected:
        inp = img.unsqueeze(0).to(DEVICE)
        orig_pred = model(inp).argmax(dim=1).item()
        h, w = img.shape[1], img.shape[2]
        plt.imshow(img.squeeze(0), cmap='gray')
        plt.title(f"GT:{label} Pred:{orig_pred}")
        plt.axis('off')
        (run_dir / "base_images").mkdir(parents=True, exist_ok=True)
        plt.savefig(run_dir / "base_images" / f"orig_id{idx}_pred{orig_pred}.png")
        plt.close()

        recursive_zero_zones(model, img, orig_pred, idx, label, run_dir,
                            0, w, 0, h, level=1, max_level=max_levels, records=xai_records)

    # Salvar summary.csv
    df_base = pd.DataFrame([{
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_ids': [r['image_id'] for r in xai_records if r['level']==1 and r['zone']=='top_left'],
        'accuracy': final_acc
    }])
    df_base.to_csv(run_dir / "summary.csv", index=False)

    # Criar DataFrame de registros, ordenar e salvar zero_zones_xai.csv
    df_xai = pd.DataFrame(xai_records)
    df_xai = df_xai.sort_values(by='level', ascending=False)
    df_xai.to_csv(run_dir / "zero_zones_xai.csv", index=False)
        
    # Agrupar records por image_id
    records_per_image = defaultdict(list)
    for rec in xai_records:
        records_per_image[rec['image_id']].append(rec)
    
    # --- Dentro do loop: gerar overlay com anotações de zone_pred ---
    for image_id, recs in records_per_image.items():
        heatmap = generate_heatmap(recs, h, w)
        # Normalização (0–1) e escala fixa
        heatmap_norm = heatmap.astype(float)
        if heatmap.max() > 0:
            heatmap_norm /= heatmap.max()

        # Carregar a imagem original
        img_tensor = [img for img, _, idx in selected if idx == image_id][0]
        img_np = img_tensor.squeeze(0).cpu().numpy()

        plt.figure(figsize=(4,4))
        # Imagem de fundo
        plt.imshow(img_np, cmap='gray', interpolation='nearest', alpha=1.0)
        # Heatmap sobreposto
        plt.imshow(heatmap_norm, cmap='jet', interpolation='nearest',
                alpha=0.6, vmin=0, vmax=1)

        # Para cada record que mudou, anotar o zone_pred no centro da região
        for rec in recs:
            if rec['changed']:
                # extrai coords e calcula centro
                r0, r1, c0, c1 = parse_coords(rec['coords'])
                center_row = (r0 + r1) / 2
                center_col = (c0 + c1) / 2
                plt.text(center_col, center_row,
                        str(rec['zone_pred']),
                        ha='center', va='center',
                        color='white',
                        fontsize='small',
                        fontweight='bold')

        plt.colorbar(label='Importância (normalizada)')
        plt.title(f"Overlay Heatmap - Image ID {image_id}")
        plt.axis('off')

        # Salvando
        heatmaps_dir = run_dir / "heatmaps_overlay"
        heatmaps_dir.mkdir(exist_ok=True)
        plt.savefig(heatmaps_dir / f"overlay_heatmap_id{image_id}.png", bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()