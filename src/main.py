import argparse
import torch
from torch.utils.data import DataLoader
import heatmap as hm
import utils as ut
import yaml
from pathlib import Path

def main():
    # --- Configuração do Parser de Argumentos ---
    parser = argparse.ArgumentParser(
        description="Executa análises de interpretabilidade (XAI) em modelos de imagem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Argumentos Principais ---
    parser.add_argument('--heatmap_type', type=str, required=True,
                        choices=['occlusion', 'zero_zone', 'scaled_zero_zone', 'knn', 'pixel_attack', 'aggregate_only'],
                        help='O tipo de análise a ser executada.')
    parser.add_argument('--model_name', type=str,
                        choices=['resnet18', 'mobilenet_v2', 'vgg16', 'squeezenet1_1'],
                        help='(Obrigatório para geração de heatmaps) Modelo SOTA a ser usado.')
    parser.add_argument('--dataset_name', type=str,
                        choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100'],
                        help='(Obrigatório para geração de heatmaps) Dataset a ser usado.')
    parser.add_argument('--save_path', type=str, default='results',
                        help='Diretório base para salvar os resultados da run.')
    
    # --- Argumentos de Geração de Heatmap ---
    gen_group = parser.add_argument_group('Geração de Heatmaps (occlusion, zero_zone, knn)')
    gen_group.add_argument('--num_images', type=int, default=1,
                           help='Número de imagens (por classe) para analisar.')
    gen_group.add_argument('--no_pretrained', action='store_true',
                           help='Se especificado, carrega a arquitetura do modelo com pesos aleatórios (sem ImageNet).')
    gen_group.add_argument('--adapt_model', action='store_true',
                           help='Se especificado, adapta a arquitetura do modelo para inputs pequenos (ex: CIFAR) em vez de redimensionar a imagem.')

    # --- Argumentos Específicos de cada Técnica ---
    occlusion_group = parser.add_argument_group('Oclusão')
    occlusion_group.add_argument('--occlusion_patch_size', type=int, default=1,
                                 help='Tamanho do patch (lado do quadrado) para a análise de oclusão.')

    zz_group = parser.add_argument_group('Zero Zones (Lento)')
    zz_group.add_argument('--max_level', type=int, default=4,
                          help='Nível máximo de recursão para Zero Zones.')
    
    knn_group = parser.add_argument_group('KNN')
    knn_group.add_argument('--knn_k', type=int, default=10, help='Número de vizinhos para KNN.')
    knn_group.add_argument('--knn_num_pixels', type=int, default=100, help='Número de pixels chave para KNN.')
    knn_group.add_argument('--knn_patch_size', type=int, default=5, help='Tamanho do patch de perturbação para KNN.')

    # --- Argumentos de Análise (Passo 2) ---
    analysis_group = parser.add_argument_group('Análise e Agregação (Passo 2)')
    analysis_group.add_argument('--npy_input_dir', type=str,
                                help='Diretório contendo os arquivos .npy para agregar ou para guiar o pixel_attack.')

    args = parser.parse_args()
    
    # --- Início da Execução ---
    base_save_path = Path(args.save_path)
    run_dir = ut.get_next_run_dir(base_save_path)

    print(f"Iniciando Run: {run_dir.name}")
    print(f"Salvando resultados em: {run_dir}")

    # Salva a configuração da run em um arquivo YAML
    config_data = vars(args)
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_data, f, indent=4, sort_keys=False)
    print(f"Configuração salva em: {run_dir / 'config.yaml'}")

    # --- Roteamento da Lógica Principal ---

    # Caso especial: agregação não precisa carregar modelo/dataset
    if args.heatmap_type == 'aggregate_only':
        if not args.npy_input_dir:
            parser.error("--npy_input_dir é obrigatório quando heatmap_type é 'aggregate_only'")
        input_path = Path(args.npy_input_dir)
        output_path = run_dir / "aggregated_plots"
        hm.aggregate_and_plot_from_npy(input_path, output_path)
        print("\nProcesso de agregação concluído com sucesso!")
        return

    # Validação para todos os outros casos que precisam de modelo e dataset
    if not args.model_name or not args.dataset_name:
        parser.error("--model_name e --dataset_name são obrigatórios para este heatmap_type.")

    # Carregamento centralizado do modelo e dataset
    print("\n--- Carregando Modelo e Dataset ---")
    model = ut.load_model(
        args.model_name, 
        args.dataset_name, 
        use_imagenet_pretrained=not args.no_pretrained, # Invertemos a lógica da flag
    )
    dataset_test, data_loader_test = ut.load_dataset(args.dataset_name, train=False)

    # Avaliação do modelo carregado
    print("\n--- Avaliando Modelo Carregado/Treinado ---")
    accuracy = ut.eval_model(model, data_loader_test)
    print(f"!!! Acurácia final do modelo no dataset de teste: {accuracy:.4f} !!!")
    if accuracy < 0.20 and args.dataset_name != 'imagenet':
        print("!!! ATENÇÃO: Acurácia muito baixa. O modelo pode não ter treinado corretamente. !!!")

    # Execução da análise de heatmap escolhida
    print(f"\n--- Executando Análise: {args.heatmap_type.upper()} ---")
    if args.heatmap_type == 'occlusion':
        hm.generate_occlusion_heatmap(model, dataset_test, run_dir,
                                      num_images_per_class=args.num_images,
                                      patch_size=args.occlusion_patch_size)
        
    elif args.heatmap_type == 'scaled_zero_zone':
        hm.generate_scaled_zero_zone_analysis(model, dataset_test, run_dir,
                                              num_images_per_class=args.num_images,
                                              max_level=args.max_level)
    elif args.heatmap_type == 'zero_zone':
        print("AVISO: O método 'zero_zone' recursivo pode ser muito lento.")
        hm.generate_zero_zone_analysis(model, dataset_test, run_dir, 
                                       num_images_per_class=args.num_images, 
                                       max_level=args.max_level)
    
    elif args.heatmap_type == 'knn':
        hm.generate_knn_heatmap(model, dataset_test, run_dir, 
                                num_images_per_class=args.num_images,
                                k_neighbors_to_paint=args.knn_k,
                                num_key_pixels_to_evaluate=args.knn_num_pixels,
                                perturb_patch_size=args.knn_patch_size)

    elif args.heatmap_type == 'pixel_attack':
        if not args.npy_input_dir:
            parser.error("--npy_input_dir é obrigatório para 'pixel_attack'")
        aggregated_heatmaps_dir = Path(args.npy_input_dir)
        hm.generate_pixel_attack_report(model, dataset_test, aggregated_heatmaps_dir, run_dir, args.num_images)
    
    else:
        # Este caso não deve ser atingido devido às 'choices' do parser, mas é uma boa prática
        raise ValueError(f"Tipo de heatmap '{args.heatmap_type}' não reconhecido no fluxo principal.")

    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()