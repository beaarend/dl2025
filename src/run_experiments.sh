#!/bin/bash

# Faz o script parar imediatamente se um comando falhar.
set -e

# --- PARÂMETROS PARA TESTAR ---
# Pares de Modelo/Dataset que serão executados.
# NOTA: A sintaxe do array foi corrigida. Cada item deve ser um elemento separado.
MODELS=("simple_cnn" "cifar_cnn" "gtsrb_cnn")
DATASETS=("mnist" "cifar10" "gtsrb")

# --- PARÂMETROS FIXOS ---
# Altere estes valores se precisar de um valor diferente para todos os testes.
NUM_IMAGES=1000
MAX_LEVEL=1000
ATTACK_PATCH_SIZE=1

# --- VERIFICAÇÃO DE CONSISTÊNCIA ---
# Garante que o número de modelos é igual ao número de datasets.
if [ ${#MODELS[@]} -ne ${#DATASETS[@]} ]; then
  echo "ERRO: O número de modelos (${#MODELS[@]}) não corresponde ao número de datasets (${#DATASETS[@]})."
  echo "Por favor, corrija os arrays no script."
  exit 1
fi

# Garante que o diretório de logs exista
echo "Criando o diretório 'logs' se não existir..."
mkdir -p logs

echo "Iniciando a execução dos testes..."
echo ""

# --- LOOP DE EXECUÇÃO SOBRE PARES VÁLIDOS ---
# Este loop itera sobre os índices dos arrays (0, 1, 2, ...).
for i in "${!MODELS[@]}"; do
  # Pega o modelo e o dataset correspondentes ao mesmo índice.
  model="${MODELS[i]}"
  dataset="${DATASETS[i]}"
    
  echo "================================================================="
  echo "INICIANDO TESTE: Modelo=[$model], Dataset=[$dataset]"
  echo "================================================================="

  # Define um nome de arquivo de log único para esta combinação.
  # A variável ${method} foi removida, pois não está mais em um loop.
  LOG_FILE="logs/log_${model}_${dataset}.txt"

  # Monta e executa o comando, redirecionando a saída padrão (stdout) e de erro (stderr)
  # para o arquivo de log.
  python3 main.py \
    --model_name "$model" \
    --dataset_name "$dataset" \
    --num_images "$NUM_IMAGES" \
    --max_level "$MAX_LEVEL" \
    --attack_patch_size "$ATTACK_PATCH_SIZE" \
    > "$LOG_FILE" 2>&1
  
  # Verifica se o comando anterior (python3) foi executado com sucesso
  if [ $? -eq 0 ]; then
    echo "SUCESSO: Teste concluído. Log salvo em: $LOG_FILE"
  else
    echo "ERRO: O teste falhou. Verifique o log para mais detalhes: $LOG_FILE"
  fi

  echo "" # Adiciona uma linha em branco para melhor legibilidade

done # Fim do loop 'for i'

echo "Todos os testes foram concluídos."