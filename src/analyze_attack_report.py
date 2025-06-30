import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Tenta importar a função de utilidade para criar runs.
# Isso torna o script autônomo, mas ainda integrado ao projeto.
try:
    from utils import get_next_run_dir
except ImportError:
    print("AVISO: 'utils.py' não encontrado. A função 'get_next_run_dir' não está disponível.")
    print("Os resultados serão salvos diretamente no diretório especificado em --save_path.")
    # Fallback: cria o diretório se a função de utilidade não existir.
    def get_next_run_dir(base_dir: Path):
        base_dir.mkdir(exist_ok=True)
        return base_dir

def analyze_confusion(df: pd.DataFrame, output_dir: Path):
    """Gera a matriz de confusão dos ataques bem-sucedidos."""
    print("\n--- 1. Análise de Confusão: Para onde as classes mudam? ---")
    successful_attacks = df[df['true_label'] != df['final_prediction']]
    
    if successful_attacks.empty:
        print("Nenhum ataque bem-sucedido registrado. Não é possível gerar a matriz de confusão.")
        return

    confusion_matrix_counts = pd.crosstab(
        successful_attacks['true_label'], 
        successful_attacks['final_prediction'],
        rownames=['Classe Verdadeira'], 
        colnames=['Classe Predita (Após Ataque)']
    )
    
    confusion_matrix_percent = confusion_matrix_counts.div(
        confusion_matrix_counts.sum(axis=1) + 1e-9, axis=0
    ) * 100
    
    common_confusions = confusion_matrix_counts.stack().sort_values(ascending=False)
    
    print("Transições de classe mais comuns durante o ataque (contagem absoluta):")
    print(common_confusions[common_confusions > 0].head(10))

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        confusion_matrix_percent, 
        annot=True, 
        cmap="YlGnBu", 
        fmt='.1f',
        cbar_kws={'label': '% de Ataques para a Classe Predita'}
    )
    plt.title("Matriz de Confusão de Ataques de Pixel (% por Classe Verdadeira)")
    plt.savefig(output_dir / "attack_confusion_matrix_percent.png", bbox_inches='tight')
    plt.close()
    print(f" -> Gráfico salvo em: {output_dir / 'attack_confusion_matrix_percent.png'}")

def analyze_robustness(df: pd.DataFrame, output_dir: Path):
    """Analisa e plota a robustez média por classe."""
    print("\n--- 2. Análise de Robustez: Qual classe é mais forte/fraca? ---")
    avg_pixels_per_class = df.groupby('true_label')['pixels_changed_percent'].mean().sort_values()
    
    most_fragile_class = avg_pixels_per_class.index[0]
    fragile_value = avg_pixels_per_class.iloc[0]
    
    most_robust_class = avg_pixels_per_class.index[-1]
    robust_value = avg_pixels_per_class.iloc[-1]
    
    print(f"Classe MAIS FRÁGIL (requer menos pixels para mudar): Classe {most_fragile_class} (média de {fragile_value:.2f}% dos pixels)")
    print(f"Classe MAIS ROBUSTA (requer mais pixels para mudar): Classe {most_robust_class} (média de {robust_value:.2f}% dos pixels)")
    
    plt.figure(figsize=(12, 6))
    avg_pixels_per_class.plot(kind='bar', color='skyblue')
    plt.title("Robustez Média por Classe (% de Pixels Necessários para Mudar Predição)")
    plt.ylabel("% Média de Pixels Modificados")
    plt.xlabel("Classe Verdadeira")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.savefig(output_dir / "average_robustness_per_class.png", bbox_inches='tight')
    plt.close()
    print(f" -> Gráfico salvo em: {output_dir / 'average_robustness_per_class.png'}")

def analyze_susceptibility(df: pd.DataFrame, output_dir: Path):
    """Analisa a suscetibilidade a ataques de poucos pixels."""
    print("\n--- 3. Análise de 'Few-Pixel Attack' (< 5% dos pixels) ---")
    few_pixel_threshold = 5.0
    few_pixel_attacks = df[df['pixels_changed_percent'] < few_pixel_threshold]
    
    if few_pixel_attacks.empty:
        print("Nenhum ataque com menos de 5% dos pixels foi registrado.")
        return

    susceptible_counts = few_pixel_attacks['true_label'].value_counts()
    total_counts = df['true_label'].value_counts()
    
    susceptibility_rate = (susceptible_counts / total_counts * 100).fillna(0).sort_values(ascending=False)
    
    most_susceptible_class = susceptibility_rate.index[0]
    print(f"Classe MAIS SUSCETÍVEL a ataques com < {few_pixel_threshold}% de pixels: Classe {most_susceptible_class}")
    print("\nTaxa de suscetibilidade por classe (% de amostras que mudaram com < 5% de pixels):")
    print(susceptibility_rate.to_string(float_format="%.2f%%"))

    plt.figure(figsize=(12, 6))
    susceptibility_rate.plot(kind='bar', color='salmon')
    plt.title(f"Taxa de Suscetibilidade a 'Few-Pixel Attack' (<{few_pixel_threshold}%)")
    plt.ylabel("% de Amostras Vulneráveis")
    plt.xlabel("Classe Verdadeira")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.savefig(output_dir / "few_pixel_attack_susceptibility.png", bbox_inches='tight')
    plt.close()
    print(f" -> Gráfico salvo em: {output_dir / 'few_pixel_attack_susceptibility.png'}")

def analyze_distribution(df: pd.DataFrame, output_dir: Path):
    """Gera um boxplot para mostrar a distribuição da robustez."""
    print("\n--- 4. Análise Adicional: Distribuição da Robustez por Classe ---")
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='true_label', y='pixels_changed_percent', palette='viridis')
    sns.stripplot(data=df, x='true_label', y='pixels_changed_percent', color='0.25', size=3, alpha=0.5)
    plt.title("Distribuição da Robustez por Classe")
    plt.ylabel("% de Pixels Modificados")
    plt.xlabel("Classe Verdadeira")
    plt.grid(axis='y', linestyle='--')
    plt.savefig(output_dir / "robustness_distribution_boxplot.png", bbox_inches='tight')
    plt.close()
    print("Esta visualização (boxplot) mostra a média, mediana, outliers e a variabilidade da robustez para cada classe.")
    print(f" -> Gráfico salvo em: {output_dir / 'robustness_distribution_boxplot.png'}")

def main():
    """
    Função principal que lê o CSV e chama as análises específicas.
    """
    parser = argparse.ArgumentParser(description="Analisa um relatório de ataque de pixel, criando um novo 'run' para cada execução.")
    parser.add_argument('--csv_path', type=str, required=True, help='Caminho para o arquivo pixel_attack_summary.csv')
    parser.add_argument('--save_path', type=str, default='analysis_results', help='Diretório base para salvar os runs da análise.')
    parser.add_argument('--analysis_name', type=str, default='all', choices=['all', 'confusion', 'robustness', 'susceptibility', 'distribution'], help='Análise específica a ser executada.')
    args = parser.parse_args()

    # Cria um novo diretório de execução para esta análise
    base_save_path = Path(args.save_path)
    run_dir = get_next_run_dir(base_save_path)
    print(f"--- Iniciando Análise do Relatório de Ataque de Pixel (Run: {run_dir.name}) ---")
    
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de relatório não encontrado em '{args.csv_path}'")
        return

    print(f"Relatório lido com sucesso. Gerando análises em: {run_dir}")

    if args.analysis_name == 'all' or args.analysis_name == 'confusion':
        analyze_confusion(df, run_dir)
    
    if args.analysis_name == 'all' or args.analysis_name == 'robustness':
        analyze_robustness(df, run_dir)
        
    if args.analysis_name == 'all' or args.analysis_name == 'susceptibility':
        analyze_susceptibility(df, run_dir)
        
    if args.analysis_name == 'all' or args.analysis_name == 'distribution':
        analyze_distribution(df, run_dir)

    print(f"\n--- Análise do Relatório Concluída. Resultados em: {run_dir} ---")

if __name__ == '__main__':
    main()
