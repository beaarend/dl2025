import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time

# --- Configurações Gerais ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Funções Auxiliares ---

def get_model_input_channels(model: nn.Module) -> int:
    """
    Tenta determinar os canais de entrada do modelo.
    Para modelos SOTA baseados em ImageNet, geralmente é 3.
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            return layer.in_channels
    # Se não encontrar Conv2d (improvável para SOTA), assume 3.
    print("AVISO: Nenhuma camada Conv2d encontrada. Assumindo 3 canais de entrada.")
    return 3

def get_next_run_dir(base_dir: Path):
    """Cria e retorna o próximo diretório de run."""
    base_dir.mkdir(exist_ok=True)
    runs = [d for d in base_dir.iterdir() if d.name.startswith("run") and d.is_dir()]
    ids = [int(d.name.replace("run", "")) for d in runs if d.name.replace("run", "").isdigit()]
    run_id = max(ids, default=0) + 1
    run_dir = base_dir / f"run{run_id}"
    run_dir.mkdir()
    return run_dir

def get_transforms(target_size: int = 224):
    """
    Cria transformações padrão para modelos SOTA (ImageNet-like).
    Garante 3 canais e tamanho 224x224.
    """
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), # Garante 3 canais
        imagenet_normalize
    ])

def load_dataset(dataset_name: str, train: bool, batch_size: int = 64):
    """Carrega dataset de treino OU teste com transformações padrão SOTA."""
    transform = get_transforms() # Sempre usa as transformações padrão (224x224, 3ch)

    print(f"Carregando dataset '{dataset_name}' (Train={train}) com transformações padrão SOTA...")

    if dataset_name == 'imagenet':
        dataset = datasets.ImageNet(root='data/imagenet', split='train' if train else 'val', transform=transform)
    elif dataset_name == 'fashionmnist':
        dataset = datasets.FashionMNIST(root='data/fashionmnist', train=train, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root='data/cifar100', train=train, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='data/cifar10', train=train, download=True, transform=transform)
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(root='data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset '{dataset_name}' não suportado.")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataset, data_loader

# --- Funções de Treino e Avaliação ---

def train_one_epoch(model, loader, optimizer, criterion):
    """Treina o modelo por uma época."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    for i, (imgs, labels) in enumerate(loader):
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
        
        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")

    epoch_time = time.time() - start_time
    return running_loss/total, correct/total, epoch_time

def eval_model(model, loader):
    """Avalia o modelo."""
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

def perform_fine_tuning(model, model_name, dataset_name, model_path, epochs=3): # Aumentei epochs para 3
    """Realiza o fine-tuning do modelo e o salva."""
    print(f"--- Iniciando Fine-Tuning: {model_name} em {dataset_name} ({epochs} epochs) ---")
    model.to(DEVICE)

    train_dataset, train_loader = load_dataset(dataset_name, train=True, batch_size=32)
    test_dataset, test_loader = load_dataset(dataset_name, train=False, batch_size=32)

    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Reduzi LR para fine-tuning
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        start_ep_time = time.time()
        print(f"[Epoch {ep}/{epochs}]")
        loss, acc_tr, train_time = train_one_epoch(model, train_loader, optimizer, criterion)
        acc_te = eval_model(model, test_loader)
        end_ep_time = time.time()
        
        print(f"  Epoch {ep} - Loss: {loss:.4f}, Train Acc: {acc_tr:.4f}, Test Acc: {acc_te:.4f} "
              f"| Train Time: {train_time:.1f}s, Total Time: {end_ep_time - start_ep_time:.1f}s")

        if acc_te > best_acc:
            best_acc = acc_te
            print(f"  -> Nova melhor acurácia ({best_acc:.4f})! Salvando modelo em {model_path}...")
            torch.save(model.state_dict(), model_path)

    print(f"--- Fine-Tuning Concluído. Melhor Acurácia: {best_acc:.4f} ---")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("AVISO: O modelo fine-tunado não foi salvo (nenhuma melhoria?). Usando a última versão.")

    return model

# --- Carregamento de Modelo ---

def load_model(model_name: str, dataset_name: str):
    """Carrega um modelo SOTA PyTorch, fine-tunando se necessário."""
    model_filename = f"{model_name}_{dataset_name}_finetuned.pth"
    model_path = MODELS_DIR / model_filename

    num_classes_map = {'mnist': 10, 'cifar10': 10, 'fashionmnist': 10, 'cifar100': 100, 'imagenet': 1000}
    num_classes = num_classes_map.get(dataset_name)
    if num_classes is None:
        raise ValueError(f"Número de classes não definido para '{dataset_name}'.")

    pretrained = not model_path.exists()
    print(f"Carregando {model_name}. Pré-treinado={pretrained}")

    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        # Adicione outros modelos SOTA aqui se desejar
        raise ValueError(f"Modelo SOTA '{model_name}' não suportado.")

    if model_path.exists():
        print(f"Carregando modelo fine-tunado de: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Modelo fine-tunado não encontrado. Iniciando fine-tuning...")
        model = perform_fine_tuning(model, model_name, dataset_name, model_path)

    return model.to(DEVICE)