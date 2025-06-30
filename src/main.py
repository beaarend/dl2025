import argparse
import torch
from torchvision import models
import heatmap as hm
import utils as ut
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Executa análise XAI (Zero Zones, KNN, etc.).")
    
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['simple_cnn', 'resnet18', 'mobilenet_v2', 'vgg16', 'efficientnet_b0'],
                        help='Nome do modelo a ser usado.')
    
    parser.add_argument('--heatmap_type', type=str, required=True,
                        choices=['gradcam', 'zero_zone', 'knn'],
                        help='Tipo de heatmap a ser criado.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['imagenet', 'fashionmnist', 'cifar10', 'cifar100', 'mnist'],
                        help='Dataset a ser usado.')
    parser.add_argument('--save_path', type=str, default='results', help='Caminho base para salvar os resultados.')
    
    # --- NOVO ARGUMENTO AQUI ---
    parser.add_argument('--show_matrix', action='store_true', 
                        help='Se presente, exibe a matriz numérica do heatmap no console.')

    # Argumentos restantes
    parser.add_argument('--max_level', type=int, default=1000, help='Nível máximo de recursão para Zero Zones.')
    parser.add_argument('--num_images', type=int, default=10, help='Número de imagens por classe para analisar.')
    parser.add_argument('--use_pretrained', type=lambda x: (str(x).lower() == 'true'), default=True, help='Usar pesos pré-treinados.')
    parser.add_argument('--knn_k', type=int, default=10, help='Número de vizinhos para KNN.')
    parser.add_argument('--knn_num_pixels', type=int, default=100, help='Número de pixels para KNN.')
    parser.add_argument('--knn_patch_size', type=int, default=10, help='Tamanho do patch para KNN.')

    args = parser.parse_args()
    
    base_save_path = Path(args.save_path)
    run_dir = ut.get_next_run_dir(base_save_path)

    print(f"Iniciando Run: {run_dir.name}")
    print(f"Configurações: {vars(args)}")
    print(f"Salvando resultados em: {run_dir}")

    config_data = vars(args)
    config_data['device'] = str(ut.DEVICE)
    config_data['run_dir'] = str(run_dir)
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_data, f, indent=4, sort_keys=False)
    print(f"Configuração salva em: {run_dir / 'config.yaml'}")

    model = ut.load_model(args.model_name, args.dataset_name, use_imagenet_pretrained=args.use_pretrained)
    
    print("\nCarregando dataset de TESTE para análise XAI...")
    dataset_test, data_loader_test = ut.load_dataset(args.dataset_name, train=False)

    print("\nAvaliando o modelo final no dataset de teste...")
    accuracy = ut.eval_model(model, data_loader_test)
    print(f"\n!!! Acurácia Final do Modelo no Teste: {accuracy:.4f} !!!\n")
    if accuracy < 0.50:
        print("!!! ATENÇÃO: A acurácia final é baixa. Verifique o processo de treino. !!!")
    
    # --- Passando o novo argumento para as funções de heatmap ---
    if args.heatmap_type == 'zero_zone':
        hm.generate_zero_zone_analysis(model, dataset_test, run_dir, 
                                       num_images_per_class=args.num_images, 
                                       max_level=args.max_level,
                                       show_matrix=args.show_matrix) # <-- Passar o argumento
    elif args.heatmap_type == 'knn':
        hm.generate_knn_heatmap(model, dataset_test, run_dir, 
                                num_images_per_class=args.num_images,
                                k_neighbors_to_paint=args.knn_k,
                                num_key_pixels_to_evaluate=args.knn_num_pixels,
                                perturb_patch_size=args.knn_patch_size,
                                show_matrix=args.show_matrix) # <-- Passar o argumento
    elif args.heatmap_type == 'gradcam':
        print("GradCAM ainda não implementado.")
    else:
        raise ValueError(f"Tipo de heatmap '{args.heatmap_type}' não suportado.")

    print("\nAnálise concluída com sucesso!")

if __name__ == "__main__":
    main()