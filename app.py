import os
import json
import streamlit as st
from google.cloud import bigquery
import openai

# 1. Carica la OpenAI API Key da Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 2. Configura BigQuery con le credenziali del Service Account
creds_json = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
# Scriviamo temporaneamente il file JSON su disco in /tmp
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_key.json"
with open("/tmp/gcp_key.json", "w") as f:
    json.dump(creds_json, f)
# Inizializza il client BigQuery
bq_client = bigquery.Client(project=creds_json["project_id"])

def ask_question(natural_question: str) -> str:
    # System prompt per generare la SQL e NLG
    system = (
        "You are a BigQuery assistant.\n"
        "When you receive a natural-language question in English, you must:\n"
        "1. Generate the correct BigQuery Standard SQL query to answer it.\n"
        "2. Execute that query against the dataset.\n"
        "3. Return only a single, concise English sentence as the answer.\n"
        "4. If the query returns no rows, reply with “No data available for that query.”\n"
        "Do not include the SQL or raw data in your reply."
    )
    # Context: schema del dataset
    context = (
        "Dataset: `planar-flux-465609-e1.locatify_data.brand_orders`\n"
        "Table `brand_orders` schema:\n"
        "- order_id STRING\n"
        "- channel STRING\n"
        "- order_date STRING (YYYY-MM-DD)\n"
        "- city STRING\n"
        "- total_eur FLOAT\n"
    )
    # Few-shot examples
    few_shot = """
Example 1
Question: “Which city has the most orders?”
SQL:
```sql
SELECT city, COUNT(*) AS cnt
FROM `planar-flux-465609-e1.locatify_data.brand_orders`
GROUP BY city
ORDER BY cnt DESC
LIMIT 1;
