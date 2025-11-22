# ifd-bridge-defects-detection

Detecção inteligente de falhas em pontes usando análise de dados estruturais.

## Pré-requisitos

- [uv](https://docs.astral.sh/uv/) - gerenciador de pacotes Python.

## Instalação

### 1. Instalar o uv

**Windows:**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/ifd-bridge-defects-detection.git
cd ifd-bridge-defects-detection
```

### 3. Instalar dependências

```bash
uv sync
```

## Uso

### Converter dados MATLAB para Parquet

**Dados baseline (ponte íntegra):**

```bash
uv run src/convert_matlab_baseline_data.py
```

**Dados com defeitos:**

```bash
uv run src/convert_matlab_damaged_data.py
```

## Estrutura do Projeto

```
ifd-bridge-defects-detection/
├── data/                    # Dados de entrada e saída
├── src/                     # Scripts de processamento
├── outputs/                 # Resultados processados
└── images/                  # Imagens e diagramas
```

## Dependências

- h5py: Leitura de arquivos HDF5
- mat73: Conversão de arquivos MATLAB
- numpy: Computação numérica
- pandas: Manipulação de dados
- pyarrow: Formato Parquet
- scipy: Computação científica
- tqdm: Barras de progresso