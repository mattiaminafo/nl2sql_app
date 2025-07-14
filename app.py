import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from openai import OpenAI
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Natural Language to SQL Query Interface",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

class NL2SQLQueryEngine:
    def __init__(self):
        self.dataset_name = "planar-flux-465609-e1.locatify_data.brand_orders"
        self.table_schema = [
            {"name": "order_id", "type": "STRING", "description": "ID univoco ordine"},
            {"name": "channel", "type": "STRING", "description": "Canale di vendita"},
            {"name": "order_date", "type": "STRING", "description": "Data ordine"},
            {"name": "order_date_timestamp", "type": "INTEGER", "description": "Timestamp ordine"},
            {"name": "first_name", "type": "STRING", "description": "Nome cliente"},
            {"name": "last_name", "type": "STRING", "description": "Cognome cliente"},
            {"name": "street", "type": "STRING", "description": "Via"},
            {"name": "house_no", "type": "STRING", "description": "Numero civico"},
            {"name": "postal_code", "type": "STRING", "description": "CAP"},
            {"name": "city", "type": "STRING", "description": "Città"},
            {"name": "country_code", "type": "STRING", "description": "Codice paese"},
            {"name": "email", "type": "STRING", "description": "Email cliente"},
            {"name": "total_eur", "type": "FLOAT", "description": "Totale in euro"},
            {"name": "customer_id", "type": "STRING", "description": "ID cliente"},
            {"name": "latitude", "type": "FLOAT", "description": "Latitudine"},
            {"name": "longitude", "type": "FLOAT", "description": "Longitudine"},
            {"name": "inserted_at", "type": "TIMESTAMP", "description": "Data inserimento"}
        ]
        self.bq_client = None
        self.openai_client = None
        self.setup_clients()

    def setup_clients(self):
        # BigQuery client
        try:
            project = st.secrets.get("project_id", "planar-flux-465609-e1")
            if "gcp_service_account" in st.secrets:
                creds = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                self.bq_client = bigquery.Client(
                    project=project,
                    credentials=creds
                )
            else:
                self.bq_client = bigquery.Client(project=project)
        except Exception as e:
            logger.error(f"BigQuery init error: {e}")
            st.error("Errore di configurazione BigQuery, controlla secrets.")

        # OpenAI client
        try:
            if "openai_api_key" in st.secrets:
                self.openai_client = OpenAI(api_key=st.secrets["openai_api_key"])
            else:
                st.error("OpenAI API key non trovata in secrets.")
        except Exception as e:
            logger.error(f"OpenAI init error: {e}")
            st.error("Errore di configurazione OpenAI, controlla secrets.")

    def generate_sql_from_nl(self, natural_language_query: str) -> str | None:
        if not self.openai_client:
            st.error("OpenAI client non inizializzato.")
            return None

        schema_desc = "\n".join(
            f"- {c['name']} ({c['type']}): {c['description']}"
            for c in self.table_schema
        )
        prompt = (
            f"You are a SQL expert. Convert the following natural language query to a valid BigQuery SQL query.\n\n"
            f"Table: {self.dataset_name}\n"
            f"Schema:\n{schema_desc}\n\n"
            "Rules:\n"
            "1. Use only the columns from the schema above\n"
            "2. Return only the SQL query, no explanations\n"
            "3. Use proper BigQuery syntax\n"
            "4. For date filtering, use the order_date column (STRING format)\n"
            "5. Always include LIMIT 100 to avoid large results\n"
            "6. Use aggregate functions when appropriate (COUNT, SUM, AVG, etc.)\n"
            "7. For month filtering, use LIKE '%2024-07%' format for July 2024\n\n"
            f"Natural Language Query: {natural_language_query}\n\n"
            "SQL Query:\n"
        )
        try:
            resp = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a SQL expert that converts natural language to BigQuery SQL."},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            sql = resp.choices[0].message.content
            return re.sub(r'```(?:sql)?\n?', '', sql).strip()
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            st.error(f"OpenAI API Error: {e}")
            return None

    def execute_sql_query(self, sql_query: str) -> tuple[pd.DataFrame | None, str | None]:
        if not self.bq_client:
            return None, "BigQuery client non inizializzato."

        if self.dataset_name not in sql_query:
            return None, "La query deve referenziare il dataset corretto."
        try:
            job = self.bq_client.query(sql_query)
            df = job.result().to_dataframe()
            if df.empty:
                return df, "Nessun risultato trovato."
            return df, None
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None, f"Errore esecuzione query: {e}"

    def format_results_to_natural_language(self, df: pd.DataFrame, original_query: str) -> str:
        if df is None or df.empty:
            return "Nessun risultato trovato."

        # single-value
        if len(df.columns) == 1 and len(df) == 1:
            col = df.columns[0]
            val = df.iloc[0, 0]
            if "count" in col.lower() or "total" in col.lower():
                return f"The result is {val:,.0f}"
            if "avg" in col.lower() or "average" in col.lower():
                return f"The average value is {val:,.2f}"
            return f"The result is {val}"

        # two-column top result
        if len(df.columns) == 2:
            c1, c2 = df.columns
            top = df.iloc[0]
            q = original_query.lower()
            if "city" in q:
                return f"The city with the most orders is {top[c1]} with {top[c2]:,.0f} orders"
            if "country" in q:
                return f"The country with the highest value is {top[c1]} with {top[c2]:,.2f}"
            if "channel" in q:
                return f"The channel with the most orders is {top[c1]} with {top[c2]:,.0f} orders"
            return f"The top result is {top[c1]} with {top[c2]}"      

        # multiple rows
        if len(df) <= 5:
            text = "Here are the results:\n"
            for _, row in df.iterrows():
                text += "• " + ", ".join(f"{c}: {row[c]}" for c in df.columns) + "\n"
            return text

        return f"Found {len(df)} results. Here are the top 5:\n" + df.head().to_string(index=False)

def main():
    st.title("🔍 Natural Language to SQL Query Interface")
    st.markdown("Ask questions about your brand orders data in plain English!")

    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = NL2SQLQueryEngine()

    with st.sidebar:
        st.header("📝 Example Questions")
        st.markdown("""
        - Which city has the most orders?
        - Which channel does the most orders in July?
        - What is the average order value?
        - How many orders were placed last month?
        - Which country has the highest total revenue?
        - Show me the top 5 cities by order count
        """)
        if st.session_state.query_history:
            st.header("🕐 Recent Queries")
            for i, q in enumerate(st.session_state.query_history[-5:]):
                st.text(f"{i+1}. {q[:50]}...")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input(
            "Enter your question in English:",
            placeholder="e.g., Which city has the most orders?"
        )
    with col2:
        ask_button = st.button("Ask Question", type="primary")

    if ask_button and user_query:
        st.session_state.query_history.append(user_query)
        with st.spinner("Processing your question..."):
            st.info("🤖 Converting your question to SQL...")
            sql = st.session_state.query_engine.generate_sql_from_nl(user_query)
            if sql:
                with st.expander("Generated SQL Query"):
                    st.code(sql, language="sql")
                st.info("🔍 Executing query on BigQuery...")
                df, error = st.session_state.query_engine.execute_sql_query(sql)
                if error:
                    st.error(f"Query Error: {error}")
                else:
                    resp = st.session_state.query_engine.format_results_to_natural_language(df, user_query)
                    st.success("✅ Query executed successfully!")
                    st.markdown("### 📊 Answer:")
                    st.markdown(f"**{resp}**")
                    if st.checkbox("Show raw data"):
                        st.dataframe(df)
            else:
                st.error("Non sono riuscito a generare una query SQL valida. Riprova a riformulare.")

    st.markdown("---")
    st.markdown("*Powered by OpenAI GPT-3.5 and Google BigQuery*")

if __name__ == "__main__":
    main()
