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
    parser.add_argument('--model_name', type=str, required=True, # Tornar obrigatório
                        choices=['resnet18', 'mobilenet_v2'],
                        help='Nome do modelo a ser usado.')
    parser.add_argument('--heatmap_type', type=str, required=True, # Tornar obrigatório
                        choices=['gradcam', 'zero_zone'],
                        help='Tipo de heatmap a ser criado.')
    parser.add_argument('--dataset_name', type=str, required=True, # Tornar obrigatório
                        choices=['imagenet', 'fashionmnist', 'cifar10', 'cifar100', 'mnist'],
                        help='Dataset a ser usado.')
    parser.add_argument('--save_path', type=str, default='results',
                        help='Caminho base para salvar os resultados.')
    # Adicionar outros parâmetros que você queira no YAML, ex:
    parser.add_argument('--max_level', type=int, default=1000,
                        help='Nível máximo de recursão para Zero Zones.')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Número de imagens por classe para analisar.')

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

    print("Carregando dataset de TESTE para análise XAI...")
    dataset, data_loader = ut.load_dataset(args.dataset_name, train=False) 

    # Carregar o modelo
    model = ut.load_model(args.model_name, args.dataset_name)

    print("Avaliando o modelo carregado/treinado no dataset de teste...")
    # Precisa carregar o test loader com o hint correto!
    _, test_loader_for_eval = ut.load_dataset(args.dataset_name, train=False)
    accuracy = ut.eval_model(model, test_loader_for_eval)
    print(f"!!! Acurácia do Modelo Carregado: {accuracy:.4f} !!!")
    if accuracy < 0.50: # Se a acurácia for muito baixa, algo está errado
        print("!!! ATENÇÃO: Acurácia muito baixa. O modelo pode estar quebrado. Verifique o fine-tuning. !!!")

    # Criar o heatmap
    if args.heatmap_type == 'zero_zone':
        hm.generate_zero_zone_analysis(model, dataset, run_dir, 
                                       num_images_per_class=args.num_images)
    elif args.heatmap_type == 'gradcam':
        print("GradCAM ainda não implementado.")

    else:
        raise ValueError(f"Tipo de heatmap {args.heatmap_type} não suportado.")

if __name__ == "__main__":
    main()