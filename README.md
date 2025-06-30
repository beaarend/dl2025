
# Ferramenta de Análise e Visualização de Heatmaps com XAI

Este projeto é uma ferramenta de linha de comando para treinar modelos de classificação de imagem, gerar heatmaps de explicabilidade (XAI), testar a robustez do modelo através de ataques de pixel e analisar os resultados para descobrir padrões de vulnerabilidade.

O principal diferencial é seu fluxo de trabalho modular, que permite gerar dados de análise, testar hipóteses e, finalmente, agregar os resultados em relatórios e visualizações claras, facilitando a experimentação e a reprodutibilidade.

## Principais Funcionalidades

* **Múltiplas Técnicas de XAI** : Suporte para `Zero Zones` e `KNN` para entender a importância dos pixels.
* **Modelos e Datasets Flexíveis** : Inclui uma `SimpleCNN` otimizada para datasets menores (MNIST, CIFAR) e suporte para modelos SOTA.
* **Treinamento Automatizado** : Inicia um processo de fine-tuning automaticamente se um modelo treinado não for encontrado.
* **Ataque de Pixel Guiado por Saliência** : Usa os heatmaps agregados para "atacar" imagens de forma inteligente, alterando os pixels mais importantes primeiro para medir a robustez do modelo.
* **Análise de Relatórios Completa** : Um script dedicado (`analyze_attack_report.py`) gera múltiplos insights a partir dos resultados dos ataques, incluindo:
* Matriz de confusão dos ataques (para onde as classes tendem a mudar).
* Ranking de classes mais e menos robustas.
* Análise de suscetibilidade a "Few-Pixel Attacks".
* **Fluxo de Trabalho Modular em Passos** : O projeto é dividido em etapas claras, desde a geração de dados até a análise final.
* **Resultados Organizados** : Cada execução cria um diretório `runX`, mantendo os resultados de cada passo isolados e fáceis de gerenciar.

## Estrutura do Projeto

```
.
├── main.py                   # Ponto de entrada para geração de dados e ataques
├── heatmap.py                # Lógica para gerar heatmaps e executar ataques
├── analyze_attack_report.py  # Script standalone para analisar os resultados dos ataques
├── utils.py                  # Utilitários: modelos, dados, treino
├── requirements.txt          # Dependências do projeto
├── models/                   # Modelos treinados (.pth) são salvos aqui
└── results/                  # Diretório para salvar os resultados dos runs de geração/ataque
    └── runX/
        ├── individual_heatmaps_npy/
        └── pixel_attack_summary.csv
└── analysis_results/         # Diretório para salvar os runs das análises de relatório
    └── runY/
        ├── attack_confusion_matrix_percent.png
        └── ...

```

## Instalação

1. **Clone o repositório:**
   ```
   git clone <url-do-seu-repositorio>
   cd <nome-do-repositorio>

   ```
2. **Crie e ative um ambiente virtual:**
   ```
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate

   ```
3. **Instale as dependências a partir de `requirements.txt`:**
   ```
   pip install -r requirements.txt

   ```

## Como Usar: O Fluxo de Trabalho Completo

O projeto opera com um fluxo de até 4 passos sequenciais.

### **Passo 1: Gerar os Heatmaps Individuais (`.npy`)**

Primeiro, gere os dados de saliência para múltiplas imagens de um dataset. O resultado são arrays NumPy (`.npy`) que contêm os heatmaps brutos para cada classe.

**Exemplo:**

```
python main.py --model_name simple_cnn --dataset_name mnist --heatmap_type zero_zone --num_images 25

```

*Saída: Uma pasta `results/runX/individual_heatmaps_npy/` com os arquivos `heatmaps_class_0.npy`, `heatmaps_class_1.npy`, etc.*

### **Passo 2: Agregar Heatmaps (Opcional, para Visualização)**

Este passo é opcional, mas recomendado para visualizar a "importância média" dos pixels para cada classe.

**Exemplo:**

```
python main.py --heatmap_type aggregate_only --npy_input_dir results/run1/individual_heatmaps_npy

```

*Saída: Uma pasta `results/run2/aggregated_plots/` com os heatmaps médios, como `aggregated_class_0.png`.*

### **Passo 3: Executar o Ataque de Pixel**

Este é o passo crucial de teste de robustez. O script usa os heatmaps `.npy` do Passo 1 para guiar um ataque, zerando os pixels mais importantes um a um até que a classificação da imagem mude.

**Exemplo:**

```
python main.py --heatmap_type pixel_attack \
               --model_name simple_cnn \
               --dataset_name mnist \
               --npy_input_dir results/run1/individual_heatmaps_npy \
               --num_images 10

```

*Saída: Um arquivo `results/run3/pixel_attack_summary.csv` contendo o relatório detalhado de cada ataque.*

### **Passo 4: Analisar o Relatório de Ataque**

O passo final. Use o script `analyze_attack_report.py` para processar o arquivo `.csv` do Passo 3 e gerar um relatório completo com gráficos e métricas.

**Exemplo (para gerar todas as análises):**

```
python analyze_attack_report.py --csv_path results/run3/pixel_attack_summary.csv

```

**Para gerar uma análise específica (ex: apenas a matriz de confusão):**

```
python analyze_attack_report.py --csv_path results/run3/pixel_attack_summary.csv --analysis_name confusion

```

*Saída: Uma pasta `analysis_results/runY/` com os gráficos e as conclusões impressas no console.*

## Detalhes dos Arquivos

* **`main.py`** : Orquestrador principal para os Passos 1, 2 e 3. Lida com a geração de dados e a execução dos ataques.
* **`heatmap.py`** : Contém a lógica das técnicas de XAI (`Zero Zones`, `KNN`) e a implementação do `pixel_attack`.
* **`utils.py`** : Módulo de suporte com arquiteturas de modelo, carregadores de dados otimizados e rotinas de treinamento.
* **`analyze_attack_report.py`** : Script **standalone** para o Passo 4. Lê o `pixel_attack_summary.csv` e produz um conjunto rico de análises sobre a robustez e as vulnerabilidades do modelo.
