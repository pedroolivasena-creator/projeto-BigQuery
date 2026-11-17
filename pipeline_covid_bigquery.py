"""
PIPELINE COVID-19 — BigQuery → Pandas → CSV/Parquet

Fonte: `bigquery-public-data.covid19_open_data.covid19_open_data`
Projeto GCP de execução: definido pela variável de ambiente GOOGLE_CLOUD_PROJECT.

DEPENDÊNCIAS
    pip install google-cloud-bigquery pandas pyarrow db-dtypes

AUTENTICAÇÃO (escolha uma opção)
1) Desenvolvimento local:
    gcloud auth application-default login
    gcloud config set project SEU_PROJECT_ID
2) Ou configure GOOGLE_APPLICATION_CREDENTIALS apontando para uma chave JSON
   de uma conta de serviço com acesso de execução de jobs no projeto.

CONFIGURAÇÃO
    Windows PowerShell:
      $env:GOOGLE_CLOUD_PROJECT="seu-project-id"
    Linux/macOS:
      export GOOGLE_CLOUD_PROJECT="seu-project-id"

EXECUÇÃO
    python pipeline_covid_bigquery.py

NOTAS
- A tabela consultada é pública; o projeto informado é o projeto de faturamento/
  execução, não uma cópia da base pública.
- A Query 1 retorna o último total confirmado disponível por país e seleciona
  os 20 maiores.
- A Query 3 retorna a série temporal do Brasil (confirmados e óbitos acumulados).
- A disponibilidade de colunas é verificada antes da extração. Caso o esquema
  público tenha mudado, a validação interrompe o pipeline com mensagem clara.
- CSV e Parquet só são gerados após autenticação e execução bem-sucedida.
"""
from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError

SOURCE_TABLE = "bigquery-public-data.covid19_open_data.covid19_open_data"

PROJECT_ID = "meu-projeto-****"

OUTPUT_DIR = Path("output")
CSV_DIR = OUTPUT_DIR / "csv"
PARQUET_DIR = OUTPUT_DIR / "parquet"

# Colunas usadas pelas consultas. A checagem abaixo confirma sua presença.
REQUIRED_COLUMNS = {
    "date",
    "country_name",
    "cumulative_confirmed",
    "cumulative_deceased",
    "aggregation_level",
}

QUERY_TOP20 = f"""
WITH latest_by_country AS (
  SELECT
    country_name,
    date,
    SAFE_CAST(cumulative_confirmed AS INT64) AS confirmed_cases
  FROM `{SOURCE_TABLE}`
  WHERE country_name IS NOT NULL
    AND cumulative_confirmed IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY country_name
    ORDER BY date DESC
  ) = 1
)
SELECT
  country_name,
  date,
  confirmed_cases
FROM latest_by_country
ORDER BY confirmed_cases DESC
LIMIT 20
"""

QUERY_BRAZIL = f"""
SELECT
    date,
    country_name,
    SAFE_CAST(cumulative_confirmed AS INT64) AS confirmed_cases,
    SAFE_CAST(cumulative_deceased AS INT64) AS confirmed_deaths

FROM `{SOURCE_TABLE}`

WHERE country_name = 'Brazil'
    AND aggregation_level = 0
    AND date IS NOT NULL

ORDER BY date
"""


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes para snake_case ASCII e remove espaços."""
    result = df.copy()
    result.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower())).strip("_")
        for col in result.columns
    ]
    return result


def validate_schema(client: bigquery.Client) -> None:
    table = client.get_table(SOURCE_TABLE)
    available = {field.name for field in table.schema}
    missing = REQUIRED_COLUMNS - available
    if missing:
        raise ValueError(
            f"Schema incompatível. Colunas ausentes: {sorted(missing)}. "
            f"Colunas disponíveis: {sorted(available)}"
        )
    logging.info("VALIDAÇÃO schema: OK — colunas obrigatórias presentes.")


def validate_dataframe(
    df: pd.DataFrame,
    name: str,
    required: Iterable[str],
    unique_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Executa validações mínimas e registra seus resultados."""
    df = normalize_columns(df)
    required = list(required)

    missing_cols = set(required) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name}: colunas ausentes: {sorted(missing_cols)}")

    if df.empty:
        raise ValueError(f"{name}: consulta retornou zero linhas.")

    null_counts = df[required].isna().sum()
    logging.info("VALIDAÇÃO %s — nulos por coluna: %s", name, null_counts.to_dict())

    if null_counts.any():
        raise ValueError(f"{name}: existem nulos nas colunas obrigatórias.")

    if unique_columns and df.duplicated(subset=unique_columns).any():
        raise ValueError(f"{name}: foram encontradas chaves duplicadas em {unique_columns}.")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError(f"{name}: datas inválidas.")
        logging.info("VALIDAÇÃO %s — tipo date: %s", name, df["date"].dtype)

    for col in ("confirmed_cases", "confirmed_deaths"):
        if col in df.columns:

            valores_originais = df[col].notna()

            valores_convertidos = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            # Identifica valores inválidos que não eram nulos originalmente
            invalidos = valores_originais & valores_convertidos.isna()

            if invalidos.any():
                raise ValueError(
                    f"{name}: existem valores não numéricos em {col}."
                )

            df[col] = valores_convertidos.astype("Int64")

            # Verifica valores negativos, ignorando os nulos
            if (df[col].dropna() < 0).any():
                raise ValueError(
                    f"{name}: valores negativos em {col}."
                )

            logging.info(
                "VALIDAÇÃO %s — coluna %s | tipo: %s | nulos: %d | mínimo: %s",
                name,
                col,
                df[col].dtype,
                df[col].isna().sum(),
                df[col].min()
            )
    logging.info(
        "VALIDAÇÃO %s — OK | linhas=%d | colunas=%d | duplicadas=%d",
        name, len(df), len(df.columns), int(df.duplicated().sum())
    )
    return df


def save_outputs(df: pd.DataFrame, filename: str) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = CSV_DIR / f"{filename}.csv"
    parquet_path = PARQUET_DIR / f"{filename}.parquet"

    # UTF-8-SIG facilita abertura do CSV no Excel em ambiente Windows.
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    logging.info("CSV salvo: %s", csv_path.resolve())
    logging.info("Parquet salvo: %s", parquet_path.resolve())


def main() -> None:
    configure_logging()

    if not PROJECT_ID:
        raise EnvironmentError(
            "meu-projeto-**** "
            "válido antes de executar."
        )

    logging.info("Projeto de execução: %s", PROJECT_ID)
    try:
        client = bigquery.Client(project=PROJECT_ID)
        validate_schema(client)

        logging.info("Executando Query 1 — Top 20 países por casos confirmados...")
        top20 = client.query(QUERY_TOP20).to_dataframe()
        top20 = validate_dataframe(
            top20,
            "top20_paises",
            required=["country_name", "date", "confirmed_cases"],
            unique_columns=["country_name"],
        )
        print("\nAMOSTRA — TOP 20 PAÍSES")
        print(top20.head(10).to_string(index=False))
        save_outputs(top20, "top20_paises_casos_confirmados")

        logging.info("Executando Query 3 — série temporal do Brazil...")
        brazil = client.query(QUERY_BRAZIL).to_dataframe()
        brazil = validate_dataframe(
            brazil,
            "brasil",
            required=["date", "country_name"],
            unique_columns=["date", "country_name"],
        )
        if not brazil["country_name"].eq("Brazil").all():
            raise ValueError("brasil: a consulta retornou país diferente de Brazil.")
        if not brazil["date"].is_monotonic_increasing:
            raise ValueError("brasil: datas fora de ordem cronológica.")

        print("\nAMOSTRA — BRAZIL")
        print(brazil.head(10).to_string(index=False))
        save_outputs(brazil, "brasil_covid_series")

        logging.info("PIPELINE CONCLUÍDO. Bases exportadas em output/csv e output/parquet.")

    except GoogleAPICallError:
        logging.exception(
            "Falha na API BigQuery. Confira autenticação, permissões, faturamento "
            "e se o projeto está ativo."
        )
        raise


if __name__ == "__main__":
    main()
