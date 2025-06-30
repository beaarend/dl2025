import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import os
from tqdm import tqdm # Usar tqdm para monitorar o treino

# --- Configurações Gerais ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NOVA ARQUITETURA: SimpleCNN para Datasets Menores ---
class SimpleCNN(nn.Module):
    """Uma CNN simples e leve, otimizada para imagens 28x28 ou 32x32."""
    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleCNN, self).__init__()
        # Input: (batch, 1, 28, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # -> (batch, 16, 14, 14)
        )
        # Input: (batch, 16, 14, 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # -> (batch, 32, 7, 7)
        )
        # Camada totalmente conectada
        # Para 28x28 -> 32 * 7 * 7 = 1568
        # Para 32x32 -> 32 * 8 * 8 = 2048 (após 2 maxpools)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # Achatamento (flatten)
        output = self.fc(x)
        return output

# --- Funções Auxiliares ---
def get_model_input_channels(model: nn.Module) -> int:
    # (Sem alterações aqui)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            return layer.in_channels
    print("AVISO: Nenhuma camada Conv2d encontrada. Assumindo 3 canais de entrada.")
    return 3

def get_next_run_dir(base_dir: Path):
    # (Sem alterações aqui)
    base_dir.mkdir(exist_ok=True)
    runs = [d for d in base_dir.iterdir() if d.name.startswith("run") and d.is_dir()]
    ids = [int(d.name.replace("run", "")) for d in runs if d.name.replace("run", "").isdigit()]
    run_id = max(ids, default=0) + 1
    run_dir = base_dir / f"run{run_id}"
    run_dir.mkdir()
    return run_dir

# --- MUDANÇA CRÍTICA: Transformações Específicas por Dataset ---
def get_dataset_specific_transforms(dataset_name: str):
    """Retorna transformações apropriadas para cada dataset, evitando redimensionamento desnecessário."""
    if dataset_name in ['mnist', 'fashionmnist']:
        # MNIST/FashionMNIST são 28x28, 1 canal. Não precisam de redimensionamento.
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Normalização simples para 1 canal
        ])
    elif dataset_name in ['cifar10', 'cifar100']:
        # CIFAR é 32x32, 3 canais.
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    elif dataset_name == 'imagenet':
        # Apenas ImageNet recebe o tratamento completo
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Transformações para o dataset '{dataset_name}' não definidas.")

def load_dataset(dataset_name: str, train: bool, batch_size: int = 64):
    """Carrega dataset com transformações OTIMIZADAS."""
    # --- MUDANÇA AQUI ---
    transform = get_dataset_specific_transforms(dataset_name)

    print(f"Carregando dataset '{dataset_name}' (Train={train}) com transformações OTIMIZADAS...")

    dataset_map = {
        'imagenet': lambda: datasets.ImageNet(root='data/imagenet', split='train' if train else 'val', transform=transform),
        'fashionmnist': lambda: datasets.FashionMNIST(root='data/fashionmnist', train=train, download=True, transform=transform),
        'cifar100': lambda: datasets.CIFAR100(root='data/cifar100', train=train, download=True, transform=transform),
        'cifar10': lambda: datasets.CIFAR10(root='data/cifar10', train=train, download=True, transform=transform),
        'mnist': lambda: datasets.MNIST(root='data', train=train, download=True, transform=transform)
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset '{dataset_name}' não suportado.")
    
    dataset = dataset_map[dataset_name]()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
    return dataset, data_loader

# --- Funções de Treino e Avaliação (com tqdm) ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    # --- MUDANÇA AQUI: Adicionando tqdm para monitorar o treino ---
    progress_bar = tqdm(loader, desc="Treinando", unit="batch", leave=False)
    for imgs, labels in progress_bar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Atualiza a barra de progresso com a loss e acurácia
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return running_loss / len(loader), correct / total

def eval_model(model, loader):
    model.eval()
    correct, total = 0, 0
    # --- MUDANÇA AQUI: Adicionando tqdm para monitorar a avaliação ---
    progress_bar = tqdm(loader, desc="Avaliando", unit="batch", leave=False)
    with torch.no_grad():
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(acc=f"{correct/total:.4f}")
    return correct/total

def perform_fine_tuning(model, model_name, dataset_name, model_path, epochs=5, initial_lr=0.001):
    print(f"--- Iniciando Treino/Fine-Tuning: {model_name} em {dataset_name} ({epochs} epochs) ---")
    model.to(DEVICE)

    ft_batch_size = 64
    train_dataset, train_loader = load_dataset(dataset_name, train=True, batch_size=ft_batch_size)
    test_dataset, test_loader = load_dataset(dataset_name, train=False, batch_size=ft_batch_size)

    optimizer = optim.Adam(model.parameters(), lr=initial_lr) 
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for ep in range(1, epochs + 1):
        print(f"[Época {ep}/{epochs}]")
        loss, acc_tr = train_one_epoch(model, train_loader, optimizer, criterion)
        acc_te = eval_model(model, test_loader)
        
        print(f"  Fim da Época {ep} -> Loss: {loss:.4f}, Acurácia Treino: {acc_tr:.4f}, Acurácia Teste: {acc_te:.4f}")
        
        if acc_te > best_acc:
            best_acc = acc_te
            print(f"  -> Nova melhor acurácia ({best_acc:.4f})! Salvando modelo em {model_path}...")
            torch.save(model.state_dict(), model_path)
            
    print(f"--- Treino Concluído. Melhor Acurácia de Teste: {best_acc:.4f} ---")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model

# --- Carregamento de Modelo (Atualizado para SimpleCNN) ---
def load_model(model_name: str, dataset_name: str, use_imagenet_pretrained: bool = True):
    model_filename = f"{model_name}_{dataset_name}_finetuned.pth"
    model_path = MODELS_DIR / model_filename

    num_classes_map = {'mnist': 10, 'cifar10': 10, 'fashionmnist': 10, 'cifar100': 100, 'imagenet': 1000}
    num_classes = num_classes_map[dataset_name]
    
    # Define os canais de entrada com base no dataset
    input_channels = 1 if dataset_name in ['mnist', 'fashionmnist'] else 3

    model = None
    
    # --- LÓGICA ATUALIZADA ---
    if model_name == 'simple_cnn':
        print(f"Carregando modelo leve 'SimpleCNN' para '{dataset_name}'.")
        model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    else: # Modelos SOTA
        weights_arg = 'IMAGENET1K_V1' if use_imagenet_pretrained and not model_path.exists() else None
        print(f"Carregando arquitetura SOTA '{model_name}'. Pesos pré-treinados: {weights_arg is not None}")
        
        if model_name == 'resnet18':
            model = models.resnet18(weights=weights_arg)
            # Adapta a primeira camada convolucional se o input não for 3 canais
            if input_channels != 3:
                model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=weights_arg)
            if input_channels != 3:
                model.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # Adicione adaptações para outros modelos SOTA se necessário
        else:
            raise ValueError(f"Modelo SOTA '{model_name}' não suportado.")

    # Lógica de Carregamento/Treino
    if model_path.exists():
        print(f"Carregando modelo treinado de: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Modelo treinado não encontrado em {model_path}. Iniciando treino do zero...")
        model = perform_fine_tuning(model, model_name, dataset_name, model_path)

    return model.to(DEVICE)