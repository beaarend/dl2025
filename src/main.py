import argparse
import torch
from torchvision import models
import heatmap as hm
import utils as ut
import yaml # <- Importar yaml
from pathlib import Path # <- Importar Path

def main():
    parser = argparse.ArgumentParser(description="Executa análise XAI (Zero Zones ou GradCAM).") # Adicionar descrição
    # Adicionando a opção para carregar um modelo específico do PyTorch
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['resnet18', 'mobilenet_v2', 'vgg16', 'efficientnet_b0'], # Adicionados
                        help='Nome do modelo SOTA a ser usado.')
    parser.add_argument('--heatmap_type', type=str, required=True, # Tornar obrigatório
                        choices=['gradcam', 'zero_zone', 'knn'],
                        help='Tipo de heatmap a ser criado.')
    parser.add_argument('--dataset_name', type=str, required=True, # Tornar obrigatório
                        choices=['imagenet', 'fashionmnist', 'cifar10', 'cifar100', 'mnist'],
                        help='Dataset a ser usado.')
    parser.add_argument('--save_path', type=str, default='results', help='Caminho base para salvar os resultados.')

    # Adicionar outros parâmetros que você queira no YAML, ex:
    parser.add_argument('--max_level', type=int, default=1000, help='Nível máximo de recursão para Zero Zones.')
    parser.add_argument('--num_images', type=int, default=10, help='Número de imagens por classe para analisar.')
    parser.add_argument('--use_pretrained', type=lambda x: (str(x).lower() == 'true'), default=True, help='Usar pesos pré-treinados do ImageNet se nenhum modelo fine-tunado existir (True/False). Default: True')

    # KNN parameters
    parser.add_argument('--knn_k', type=int, default=10, help='Número de vizinhos para KNN (apenas para heatmap knn).')
    parser.add_argument('--knn_num_pixels', type=int, default=100, help='Número de pixels para KNN (apenas para heatmap knn).')
    parser.add_argument('--knn_patch_size', type=int, default=10, help='Tamanho do patch para KNN (apenas para heatmap knn).')

    args = parser.parse_args()
    
    # --- Criar diretório da Run ---
    base_save_path = Path(args.save_path)
    run_dir = ut.get_next_run_dir(base_save_path) # <- Criar a run_dir aqui

    print(f"Iniciando Run: {run_dir.name}")
    print(f"Configurações: {args}")
    print(f"Salvando resultados em: {run_dir}")

    # --- Salvar Configuração em YAML ---
    config_data = vars(args) # Converte args para um dicionário
    config_data['device'] = str(ut.DEVICE) # Adiciona info do device
    config_data['run_dir'] = str(run_dir) # Adiciona o próprio caminho da run

    config_yaml_path = run_dir / 'config.yaml'
    try:
        with open(config_yaml_path, 'w') as f:
            yaml.dump(config_data, f, indent=4, sort_keys=False)
        print(f"Configuração salva em: {config_yaml_path}")
    except Exception as e:
        print(f"Erro ao salvar config.yaml: {e}")
    # ------------------------------------

    # Carregar o dataset de TESTE para a análise XAI
    print("Carregando dataset de TESTE para análise XAI...")
    # A função load_dataset não precisa mais do model_name_hint, pois get_transforms foi simplificada
    dataset_test, data_loader_test = ut.load_dataset(args.dataset_name, train=False)

    # Carregar o modelo
    model = ut.load_model(args.model_name, args.dataset_name, use_imagenet_pretrained=args.use_pretrained)

    print("Avaliando o modelo carregado/treinado no dataset de teste...")
    accuracy = ut.eval_model(model, data_loader_test) # Usa o data_loader_test
    print(f"!!! Acurácia do Modelo Carregado/Fine-tunado: {accuracy:.4f} !!!")
    if accuracy < 0.20 and args.dataset_name != 'imagenet': # Ajuste o limiar conforme o dataset
        print("!!! ATENÇÃO: Acurácia muito baixa. O modelo pode não ter treinado corretamente. !!!")
    
    # ... (resto do main, chamando generate_zero_zone_analysis com dataset_test) ...
    if args.heatmap_type == 'zero_zone':
        hm.generate_zero_zone_analysis(model, dataset_test, run_dir, 
                                       num_images_per_class=args.num_images, 
                                       max_level=args.max_level)
    elif args.heatmap_type == 'knn':
        hm.generate_knn_heatmap(model, dataset_test, run_dir, 
                                num_images_per_class=args.num_images,
                                k_neighbors_to_paint=args.knn_k,
                                num_key_pixels_to_evaluate=args.knn_num_pixels,
                                perturb_patch_size=args.knn_patch_size)    
    
    # Criar o heatmap
    if args.heatmap_type == 'zero_zone':
        hm.generate_zero_zone_analysis(model, dataset_test, run_dir, 
                                       num_images_per_class=args.num_images)
    elif args.heatmap_type == 'gradcam':
        print("GradCAM ainda não implementado.")
    elif args.heatmap_type == 'knn':
        print("nao entendi pq tem essa parte aqui, mas vou deixar assim mesmo")

    else:
        raise ValueError(f"Tipo de heatmap {args.heatmap_type} não suportado.")

if __name__ == "__main__":
    main()