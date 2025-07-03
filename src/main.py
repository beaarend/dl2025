import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
import heatmap as hm
import utils as ut
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Executa um fluxo integrado de análise de interpretabilidade e ataque.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Argumentos Principais ---
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['simple_cnn', 'cifar_cnn', 'gtsrb_cnn', 'tiny_imagenet_cnn', 'resnet18', 'mobilenet_v2', 'vgg16', 'squeezenet1_1'],
                        help='Modelo a ser usado.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['mnist', 'fashionmnist', 'cifar10', 'gtsrb', 'cifar100', 'tiny_imagenet'],
                        help='Dataset a ser usado.')
    parser.add_argument('--save_path', type=str, default='results',
                        help='Diretório base para salvar os resultados da run.')
    
    # --- Argumentos de Configuração do Fluxo ---
    run_group = parser.add_argument_group('Configuração da Execução')
    run_group.add_argument('--num_images', type=int, default=20,
                           help='Número TOTAL de imagens (por classe) para o fluxo. Será dividido para gerar heatmaps e atacar.')
    run_group.add_argument('--max_level', type=int, default=4,
                          help='Nível máximo de recursão para o método Scaled Zero Zones.')
    run_group.add_argument('--attack_patch_size', type=int, default=1,
                           help='Tamanho do patch para o pixel_attack (ex: 1 para pixel, 3 para 3x3).')
    run_group.add_argument('--no_pretrained', action='store_true',
                           help='Para modelos SOTA, carrega a arquitetura com pesos aleatórios (sem ImageNet).')
    args = parser.parse_args()
    
    # --- Início da Execução ---
    base_save_path = Path(args.save_path)
    run_dir = ut.get_next_run_dir(base_save_path)
    print(f"Iniciando Run: {run_dir.name}")
    
    config_data = vars(args)
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_data, f, indent=4, sort_keys=False)
    print(f"Configuração salva em: {run_dir / 'config.yaml'}")

       # Carregamento do modelo e dataset
    print("\n--- Carregando Modelo e Dataset ---")
    model = ut.load_model(args.model_name, args.dataset_name, use_imagenet_pretrained=not args.no_pretrained)
    
    # Carrega ambos os splits do dataset
    print("Carregando split de treino...")
    dataset_train, _ = ut.load_dataset(args.dataset_name, train=True, model_name=args.model_name)
    print("Carregando split de teste...")
    dataset_test, data_loader_test = ut.load_dataset(args.dataset_name, train=False, model_name=args.model_name)

    # Combina os datasets de treino e teste em um único dataset para a análise
    print(f"Combinando datasets: {len(dataset_train)} (treino) + {len(dataset_test)} (teste) imagens.")
    full_dataset = ConcatDataset([dataset_train, dataset_test])
    print(f"Tamanho total do dataset para análise: {len(full_dataset)} imagens.")
    
    # Avaliação do modelo (continua sendo feita APENAS no de teste, o que é a prática correta)
    print("\n--- Avaliando Modelo Carregado/Treinado (no split de teste) ---")
    accuracy = ut.eval_model(model, data_loader_test)
    print(f"!!! Acurácia final do modelo no dataset de teste: {accuracy:.4f} !!!")

    class_names = ut.get_class_names(args.dataset_name)
    if class_names:
        print(f"Nomes de classe para '{args.dataset_name}' carregados.")
    
    # Execução do fluxo integrado no dataset COMPLETO
    hm.run_integrated_heatmap_attack(
        model,
        full_dataset,  # <- Passa o dataset combinado
        run_dir,
        class_names=class_names,      # <- Passa o mapa de nomes
        num_images_per_class=args.num_images,
        max_level=args.max_level,
        attack_patch_size=args.attack_patch_size,
    )

if __name__ == "__main__":
    main()