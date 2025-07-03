import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

def download_file(url, target_path):
    """Baixa um arquivo com uma barra de progresso."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Baixando {target_path.name}")
    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERRO: Algo deu errado durante o download.")

def main():
    """
    Script para baixar o Tiny ImageNet e preparar um subconjunto de 10 classes.
    """
    # --- Configuração ---
    DATA_DIR = Path("./data")
    TINY_IMAGENET_DIR = DATA_DIR / "tiny-imagenet-200"
    ZIP_PATH = DATA_DIR / "tiny-imagenet-200.zip"
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    CHOSEN_CLASSES = [
        'n02123045',  # cat
        'n02085620',  # dog
        'n01910747',  # jellyfish
        'n01770393',  # scorpion
        'n02129165',  # lion
    ]
    
    SUBSET_DIR = DATA_DIR / "tiny-imagenet-10"

    # --- Passo 1: Download e Descompactação ---
    DATA_DIR.mkdir(exist_ok=True)
    
    if not TINY_IMAGENET_DIR.exists():
        if not ZIP_PATH.exists():
            print(f"Baixando Tiny ImageNet de {URL}...")
            download_file(URL, ZIP_PATH)
        
        print(f"Descompactando {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Descompactação concluída.")
    else:
        print(f"Diretório {TINY_IMAGENET_DIR} já existe. Pulando download e descompactação.")

    # --- Passo 2: Preparação do Subconjunto ---
    if SUBSET_DIR.exists():
        print(f"Diretório do subconjunto {SUBSET_DIR} já existe. Se quiser recriá-lo, delete a pasta manualmente.")
        return

    print(f"\nCriando subconjunto de 10 classes em {SUBSET_DIR}...")
    (SUBSET_DIR / "train").mkdir(parents=True)
    (SUBSET_DIR / "val").mkdir(parents=True)

    # Processar TREINO
    print("Processando dados de TREINO...")
    for class_id in tqdm(CHOSEN_CLASSES, desc="Copiando classes de treino"):
        source_path = TINY_IMAGENET_DIR / "train" / class_id / "images"
        target_path = SUBSET_DIR / "train" / class_id
        target_path.mkdir(parents=True, exist_ok=True)

        if source_path.exists():
            for img_file in source_path.glob("*.JPEG"):
                shutil.copy(img_file, target_path / img_file.name)

    # Processar VALIDAÇÃO
    print("Processando dados de VALIDAÇÃO...")
    val_annotations_path = TINY_IMAGENET_DIR / "val" / "val_annotations.txt"
    val_images_path = TINY_IMAGENET_DIR / "val" / "images"

    val_count = 0
    with open(val_annotations_path, 'r') as f:
        for line in tqdm(f.readlines(), desc="Copiando imagens de validação"):
            parts = line.strip().split('\t')
            img_name, class_id = parts[0], parts[1]
            
            if class_id in CHOSEN_CLASSES:
                class_val_dir = SUBSET_DIR / "val" / class_id
                class_val_dir.mkdir(parents=True, exist_ok=True)

                source_img_path = val_images_path / img_name
                target_img_path = class_val_dir / img_name
                if source_img_path.exists():
                    shutil.copyfile(source_img_path, target_img_path)
                    val_count += 1

    print(f"\nTotal de imagens de validação copiadas: {val_count}")
    
    # --- Verificação final ---
    print("\nResumo final:")
    for split in ['train', 'val']:
        print(f"\n{split.upper()} SET")
        for class_id in CHOSEN_CLASSES:
            folder = SUBSET_DIR / split / class_id
            count = len(list(folder.glob("*.JPEG"))) if folder.exists() else 0
            print(f"  - Classe {class_id}: {count} imagens")

    print(f"\nSubconjunto criado com sucesso em '{SUBSET_DIR}'!")

if __name__ == "__main__":
    main()
