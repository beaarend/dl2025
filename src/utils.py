import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import os
from tqdm import tqdm

# --- Configurações Gerais ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ARQUITETURAS CUSTOMIZADAS ---
class SimpleCNN(nn.Module):
    """Uma CNN simples e leve, flexível para imagens 28x28 ou 32x32."""
    def __init__(self, num_classes=10, input_channels=1, input_size=28):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 28x28 -> 14x14 | 32x32 -> 16x16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 14x14 -> 7x7 | 16x16 -> 8x8
        )
        final_conv_size = input_size // 4
        flattened_features = 32 * final_conv_size * final_conv_size
        self.fc = nn.Linear(flattened_features, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CIFAR_CNN(nn.Module):
    """Uma CNN otimizada para datasets 32x32 como o CIFAR-10."""
    def __init__(self, num_classes=10):
        super(CIFAR_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class GTSRB_CNN(nn.Module):
    """Uma CNN projetada para o GTSRB (redimensionado para 48x48)."""
    def __init__(self, num_classes=43):
        super(GTSRB_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 48 -> 24
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 24 -> 12
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_class_names(dataset_name: str):
    """Retorna um dicionário mapeando índices de classe para nomes legíveis."""
    if dataset_name == 'cifar10':
        return {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }
    if dataset_name == 'mnist':
        return {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
        }
    if dataset_name == 'gtsrb':
        # Nomes resumidos para melhor visualização nos plots
        return {
            0: '20 km/h', 1: '30 km/h', 2: '50 km/h', 3: '60 km/h', 4: '70 km/h',
            5: '80 km/h', 6: 'Fim 80', 7: '100 km/h', 8: '120 km/h', 9: 'Proib. Ultrapassar',
            10: 'Proib. P/ >3.5t', 11: 'Prioridade Cruzamento', 12: 'Via Prioritária', 13: 'Dê a Preferência',
            14: 'PARE', 15: 'Proibido Veículos', 16: 'Proibido >3.5t', 17: 'Proibido Entrar',
            18: 'Cuidado Geral', 19: 'Curva Perigosa (E)', 20: 'Curva Perigosa (D)', 21: 'Curva Dupla',
            22: 'Pista Irregular', 23: 'Pista Escorregadia', 24: 'Estreitamento (D)', 25: 'Obras',
            26: 'Semáforo', 27: 'Pedestres', 28: 'Crianças', 29: 'Ciclistas',
            30: 'Neve/Gelo', 31: 'Animais Selvagens', 32: 'Fim de Proibições', 33: 'Vire à Direita',
            34: 'Vire à Esquerda', 35: 'Siga em Frente', 36: 'Frente ou Direita', 37: 'Frente ou Esquerda',
            38: 'Mantenha-se à Direita', 39: 'Mantenha-se à Esquerda', 40: 'Rotatória', 41: 'Fim Proib. Ultrapassar',
            42: 'Fim Proib. >3.5t'
        }
    # Para MNIST e outros, retorna None para que o código continue usando os números
    return None

# --- Funções Auxiliares ---
def get_next_run_dir(base_dir: Path):
    """Cria e retorna o próximo diretório de run."""
    base_dir.mkdir(exist_ok=True)
    runs = [d for d in base_dir.iterdir() if d.name.startswith("run") and d.is_dir()]
    ids = [int(d.name.replace("run", "")) for d in runs if d.name.replace("run", "").isdigit()]
    run_id = max(ids, default=0) + 1
    run_dir = base_dir / f"run{run_id}"
    run_dir.mkdir()
    return run_dir

def print_trainable_parameters(model):
    """Imprime o número de parâmetros treináveis e o total."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  -> Parâmetros Treináveis: {trainable_params:,} | "
        f"Total de Parâmetros: {total_params:,} | "
        f"Ratio Treinável: {100 * trainable_params / total_params:.2f}%"
    )

# --- Carregamento de Dados e Transformações ---
def get_transforms(model_name: str, dataset_name: str):
    """Retorna as transformações corretas baseadas no modelo e no dataset."""
    is_sota_model = model_name not in ['simple_cnn', 'cifar_cnn', 'gtsrb_cnn']
    
    if is_sota_model:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    if model_name == 'simple_cnn':
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    if model_name == 'cifar_cnn':
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    
    if model_name == 'gtsrb_cnn':
        # GTSRB tem tamanhos variados, então redimensionamos para 48x48 para nossa CNN customizada
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]) # Médias específicas do GTSRB
        ])
    
    # Fallback genérico se nenhuma condição for atendida
    return transforms.Compose([transforms.ToTensor()])

def load_dataset(dataset_name: str, train: bool, batch_size: int = 64, model_name: str = "resnet18"):
    """Carrega dataset com transformações baseadas no modelo."""
    transform = get_transforms(model_name, dataset_name)
    print(f"Carregando dataset '{dataset_name}' (Train={train}) para o modelo '{model_name}'...")
    
    if dataset_name == 'gtsrb':
        dataset = datasets.GTSRB(root='data/gtsrb', split='train' if train else 'test', download=True, transform=transform)
    else:
        dataset_map = {
            'fashionmnist': datasets.FashionMNIST, 'cifar100': datasets.CIFAR100,
            'cifar10': datasets.CIFAR10, 'mnist': datasets.MNIST
        }
        if dataset_name not in dataset_map:
            raise ValueError(f"Dataset '{dataset_name}' não suportado.")
        dataset = dataset_map[dataset_name](root=f'data/{dataset_name}', train=train, download=True, transform=transform)
        
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return dataset, data_loader


# --- Funções de Treino e Avaliação ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    return running_loss / len(loader), correct / total

def eval_model(model, loader):
    model.eval()
    correct, total = 0, 0
    progress_bar = tqdm(loader, desc="Avaliando", unit="batch", leave=False)
    with torch.no_grad():
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(acc=f"{correct/total:.4f}")
    return correct / total

def perform_training(model, model_name, dataset_name, model_path, epochs=50, initial_lr=0.001):
    print(f"--- Iniciando Treinamento: {model_name} em {dataset_name} ({epochs} epochs) ---")
    model.to(DEVICE)

    train_dataset, train_loader = load_dataset(dataset_name, train=True, batch_size=32, model_name=model_name)
    test_dataset, test_loader = load_dataset(dataset_name, train=False, batch_size=32, model_name=model_name)

    params_to_update = [param for param in model.parameters() if param.requires_grad]
    if not params_to_update:
        raise ValueError("Nenhum parâmetro para treinar!")

    optimizer = optim.Adam(params_to_update, lr=initial_lr)
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

    print(f"--- Treinamento Concluído. Melhor Acurácia: {best_acc:.4f} ---")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model

# --- Carregamento de Modelo ---
def load_model(model_name: str, dataset_name: str, use_imagenet_pretrained: bool = True):
    is_sota_model = model_name not in ['simple_cnn', 'cifar_cnn', 'gtsrb_cnn']
    
    model_filename = f"{model_name}_{dataset_name}_trained.pth"
    model_path = MODELS_DIR / model_filename

    num_classes_map = {'mnist': 10, 'cifar10': 10, 'fashionmnist': 10, 'cifar100': 100, 'gtsrb': 43}
    num_classes = num_classes_map.get(dataset_name)
    if num_classes is None: raise ValueError(f"Número de classes não definido para '{dataset_name}'.")

    model = None
    
    if model_name == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == 'cifar_cnn':
        model = CIFAR_CNN(num_classes=num_classes)
    elif model_name == 'gtsrb_cnn':
        model = GTSRB_CNN(num_classes=num_classes)
    elif is_sota_model:
        load_initial_imagenet_weights = use_imagenet_pretrained and not model_path.exists()
        weights_arg = 'IMAGENET1K_V1' if load_initial_imagenet_weights else None
        print(f"Carregando arquitetura SOTA '{model_name}'. Pesos pré-treinados: {weights_arg is not None}")
        
        if model_name == 'resnet18':
            model = models.resnet18(weights=weights_arg)
            for param in model.parameters(): param.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        # ... (outros modelos SOTA) ...
        else:
            raise ValueError(f"Modelo SOTA '{model_name}' não suportado.")
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")

    if model_path.exists():
        print(f"Carregando modelo treinado de: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Modelo treinado não encontrado. Iniciando treinamento...")
        model = perform_training(model, model_name, dataset_name, model_path)
    
    print("Contagem de parâmetros do modelo final:")
    print_trainable_parameters(model)
    
    return model.to(DEVICE)
