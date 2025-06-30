import argparse
import torch
import heatmap as hm
import utils as ut
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Executa análise XAI ou agregação de heatmaps.")
    
    # --- MUDANÇA AQUI: Adicionando 'aggregate_only' ---
    parser.add_argument('--heatmap_type', type=str, required=True,
                        choices=['zero_zone', 'knn', 'aggregate_only', 'pixel_attack'],
                        help='Tipo de análise: "zero_zone" ou "knn" para gerar .npy, "aggregate_only" para processá-los.')
    
    # --- NOVO ARGUMENTO para o Passo 2 ---
    parser.add_argument('--npy_input_dir', type=str,
                        help='(Obrigatório para aggregate_only) Diretório contendo os arquivos .npy para agregar.')

    # Argumentos que se aplicam a ambos os passos
    parser.add_argument('--model_name', type=str, help='Modelo a ser usado.',
                    choices=['simple_cnn', 'cifar_cnn', 'resnet18', 'mobilenet_v2', 'squeezenet1_1'])
    parser.add_argument('--dataset_name', type=str, help='(Obrigatório para geração) Dataset a ser usado.')
    parser.add_argument('--save_path', type=str, default='results', help='Caminho base para salvar os resultados.')
    parser.add_argument('--num_images', type=int, default=20, help='Número de imagens (por classe) para analisar.')
    
    # Argumentos específicos de cada técnica
    parser.add_argument('--max_level', type=int, default=1000, help='Nível máximo para Zero Zones.')
    parser.add_argument('--knn_k', type=int, default=10, help='Número de vizinhos para KNN.')
    parser.add_argument('--knn_num_pixels', type=int, default=100, help='Número de pixels para KNN.')
    parser.add_argument('--knn_patch_size', type=int, default=10, help='Tamanho do patch para KNN.')
    
    args = parser.parse_args()
    
    base_save_path = Path(args.save_path)
    run_dir = ut.get_next_run_dir(base_save_path)

    print(f"Iniciando Run: {run_dir.name}")
    print(f"Salvando resultados em: {run_dir}")
    
    # --- LÓGICA DE ROTEAMENTO ATUALIZADA ---
    if args.heatmap_type == 'pixel_attack':
        if not all([args.model_name, args.dataset_name, args.npy_input_dir]):
            parser.error("--model_name, --dataset_name, e --npy_input_dir são obrigatórios para 'pixel_attack'")
        
        model = ut.load_model(args.model_name, args.dataset_name)
        dataset_test, _ = ut.load_dataset(args.dataset_name, train=False)
        aggregated_heatmaps_dir = Path(args.npy_input_dir)
        
        hm.generate_pixel_attack_report(model, dataset_test, aggregated_heatmaps_dir, run_dir, args.num_images)
        
    elif args.heatmap_type == 'aggregate_only':
        if not args.npy_input_dir:
            parser.error("--npy_input_dir é obrigatório quando heatmap_type é 'aggregate_only'")
        
        input_path = Path(args.npy_input_dir)
        output_path = run_dir / "aggregated_plots"
        hm.aggregate_and_plot_from_npy(input_path, output_path)
    
    elif args.heatmap_type in ['zero_zone', 'knn']:
        # Validação de argumentos para a geração
        if not args.model_name or not args.dataset_name:
            parser.error("--model_name e --dataset_name são obrigatórios para gerar heatmaps.")
            
        model = ut.load_model(args.model_name, args.dataset_name, use_imagenet_pretrained=True)
        dataset_test, _ = ut.load_dataset(args.dataset_name, train=False)
        
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
    else:
        raise ValueError("Tipo de heatmap inválido.")

    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()