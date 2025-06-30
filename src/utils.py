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

# --- ARQUITETURA SimpleCNN ATUALIZADA ---
class SimpleCNN(nn.Module):
    """Uma CNN simples e leve, agora flexível para imagens 28x28 ou 32x32."""
    def __init__(self, num_classes=10, input_channels=1, input_size=28):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Camada 1
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 28x28 -> 14x14 | 32x32 -> 16x16
            
            # Camada 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 14x14 -> 7x7 | 16x16 -> 8x8
        )
        
        # --- MUDANÇA CRÍTICA AQUI ---
        # Calcula o tamanho da saída das camadas convolucionais dinamicamente
        # Após 2 camadas de MaxPool com kernel 2, o tamanho da imagem é dividido por 4.
        final_conv_size = input_size // 4
        flattened_features = 32 * final_conv_size * final_conv_size
        
        # A camada totalmente conectada agora usa o tamanho calculado
        self.fc = nn.Linear(flattened_features, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Achatamento (flatten)
        output = self.fc(x)
        return output

# --- NOVA ARQUITETURA: CIFAR_CNN ---
class CIFAR_CNN(nn.Module):
    """Uma CNN otimizada para datasets 32x32 como o CIFAR-10."""
    def __init__(self, num_classes=10):
        super(CIFAR_CNN, self).__init__()
        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16
            nn.Dropout(0.25),

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16 -> 8x8
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
    transform = get_dataset_specific_transforms(dataset_name)
    print(f"Carregando dataset '{dataset_name}' (Train={train}) com transformações OTIMIZADAS...")
    dataset_map = {
        'imagenet': lambda: datasets.ImageNet(root='data/imagenet', split='train' if train else 'val', transform=transform),
        'fashionmnist': lambda: datasets.FashionMNIST(root='data/fashionmnist', train=train, download=True, transform=transform),
        'cifar100': lambda: datasets.CIFAR100(root='data/cifar100', train=train, download=True, transform=transform),
        'cifar10': lambda: datasets.CIFAR10(root='data/cifar10', train=train, download=True, transform=transform),
        'mnist': lambda: datasets.MNIST(root='data', train=train, download=True, transform=transform)
    }
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

def perform_fine_tuning(model, model_name, dataset_name, model_path, epochs=30, initial_lr=0.001):
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

# --- Carregamento de Modelo (Atualizado para incluir CIFAR_CNN) ---
def load_model(model_name: str, dataset_name: str, use_imagenet_pretrained: bool = True):
    model_filename = f"{model_name}_{dataset_name}_finetuned.pth"
    model_path = MODELS_DIR / model_filename

    num_classes_map = {'mnist': 10, 'cifar10': 10, 'fashionmnist': 10, 'cifar100': 100, 'imagenet': 1000}
    num_classes = num_classes_map.get(dataset_name, 10)
    
    if dataset_name in ['mnist', 'fashionmnist']:
        input_channels, input_size = 1, 28
    elif dataset_name in ['cifar10', 'cifar100']:
        input_channels, input_size = 3, 32
    else:
        input_channels, input_size = 3, 224
    
    model = None
    
    if model_name == 'simple_cnn':
        print(f"Carregando modelo 'SimpleCNN' para '{dataset_name}'.")
        model = SimpleCNN(num_classes=num_classes, input_channels=input_channels, input_size=input_size)
    
    # --- MUDANÇA AQUI: Adicionado suporte para CIFAR_CNN ---
    elif model_name == 'cifar_cnn':
        print(f"Carregando modelo 'CIFAR_CNN' otimizado para 32x32.")
        if input_channels != 3:
            print("AVISO: CIFAR_CNN é otimizado para 3 canais de entrada (RGB).")
        model = CIFAR_CNN(num_classes=num_classes)
    
    else: # Modelos SOTA
        weights_arg = 'IMAGENET1K_V1' if use_imagenet_pretrained and not model_path.exists() else None
        print(f"Carregando arquitetura SOTA '{model_name}'. Pesos pré-treinados: {weights_arg is not None}")
        
        if model_name == 'resnet18':
            model = models.resnet18(weights=weights_arg)
            if input_channels != 3: model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=weights_arg)
            if input_channels != 3: model.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        elif model_name == 'squeezenet1_1':
            model = models.squeezenet1_1(weights=weights_arg)
            if input_channels != 3: print("AVISO: SqueezeNet é otimizado para 3 canais de entrada (RGB).")
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model.num_classes = num_classes
        
        else:
            raise ValueError(f"Modelo SOTA '{model_name}' não suportado.")

    if model_path.exists():
        print(f"Carregando modelo treinado de: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Modelo treinado não encontrado em {model_path}. Iniciando treino...")
        # Aumentar épocas para o novo modelo pode ser uma boa ideia
        epochs = 15 if model_name == 'cifar_cnn' else 5
        model = perform_fine_tuning(model, model_name, dataset_name, model_path, epochs=epochs)

    return model.to(DEVICE)