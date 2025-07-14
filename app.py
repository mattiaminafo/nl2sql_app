# app.py
import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import openai
import json
import re
from datetime import datetime
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
        self.setup_clients()
    
    def setup_clients(self):
        """Initialize BigQuery and OpenAI clients"""
        try:
            # ——————————— BigQuery ———————————
            if "gcp_service_account" in st.secrets:
                credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                self.bq_client = bigquery.Client(credentials=credentials)
            elif "project_id" in st.secrets:
                self.bq_client = bigquery.Client(project=st.secrets["project_id"])
            else:
                self.bq_client = bigquery.Client(project="planar-flux-465609-e1")
            
            # ——————————— OpenAI ———————————
            if "openai_api_key" in st.secrets:
                openai.api_key = st.secrets["openai_api_key"]
            else:
                st.error("OpenAI API key not found in secrets")
                return
                
        except Exception as e:
            logger.error(f"Error setting up clients: {e}")
            st.error(f"Error setting up clients: {e}")
    
    def generate_sql_from_nl(self, natural_language_query):
        """Convert natural language to SQL using OpenAI"""
        schema_description = "\n".join([
            f"- {col['name']} ({col['type']}): {col['description']}"
            for col in self.table_schema
        ])
        
        prompt = f"""
You are a SQL expert. Convert the following natural language query to a valid BigQuery SQL query.

Table: {self.dataset_name}
Schema:
{schema_description}

Rules:
1. Use only the columns from the schema above
2. Return only the SQL query, no explanations
3. Use proper BigQuery syntax
4. For date filtering, use the order_date column (STRING format)
5. Always include LIMIT 100 to avoid large results
6. Use aggregate functions when appropriate (COUNT, SUM, AVG, etc.)
7. For month filtering, use LIKE '%2024-07%' format for July 2024

Natural Language Query: {natural_language_query}

SQL Query:
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a SQL expert that converts natural language to BigQuery SQL."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            # Rimuovi eventuali backticks
            sql_query = re.sub(r'```(?:sql)?\n?', '', sql_query).strip()
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            st.error(f"OpenAI API Error: {e}")
            return None
    
    def execute_sql_query(self, sql_query):
        """Execute SQL query on BigQuery"""
        try:
            # Validate query references the corretto dataset
            if self.dataset_name not in sql_query:
                return None, "Query must reference the correct dataset table"
            
            query_job = self.bq_client.query(sql_query)
            results = query_job.result()
            df = results.to_dataframe()
            
            if df.empty:
                return df, "No results found for your query"
            return df, None
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None, f"Error executing query: {e}"
    
    def format_results_to_natural_language(self, df, original_query):
        """Convert query results to natural language response"""
        if df is None or df.empty:
            return "No results found for your query."
        
        try:
            # Single value
            if len(df.columns) == 1 and len(df) == 1:
                col = df.columns[0]
                val = df.iloc[0, 0]
                if "count" in col.lower() or "total" in col.lower():
                    return f"The result is {val:,.0f}"
                if "avg" in col.lower() or "average" in col.lower():
                    return f"The average value is {val:,.2f}"
                return f"The result is {val}"
            
            # Two columns, multiple rows
            if len(df.columns) == 2 and len(df) >= 1:
                col1, col2 = df.columns
                top = df.iloc[0]
                q = original_query.lower()
                if "city" in q:
                    return f"The city with the most orders is {top[col1]} with {top[col2]:,.0f} orders"
                if "country" in q:
                    return f"The country with the highest value is {top[col1]} with {top[col2]:,.2f}"
                if "channel" in q:
                    return f"The channel with the most orders is {top[col1]} with {top[col2]:,.0f} orders"
                return f"The top result is {top[col1]} with {top[col2]}"
            
            # Multiple columns or rows
            if len(df) <= 5:
                text = "Here are the results:\n"
                for _, row in df.iterrows():
                    text += "• " + ", ".join(f"{c}: {row[c]}" for c in df.columns) + "\n"
                return text
            return f"Found {len(df)} results. Here are the top 5:\n" + df.head().to_string(index=False)
        
        except Exception as e:
            logger.error(f"Error formatting: {e}")
            return f"Results found but formatting error: {e}"

def main():
    st.title("🔍 Natural Language to SQL Query Interface")
    st.markdown("Ask questions about your brand orders data in plain English!")
    
    # Initialize query engine
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = NL2SQLQueryEngine()
    
    # Sidebar with examples
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
    
    # Main columns
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
            sql_query = st.session_state.query_engine.generate_sql_from_nl(user_query)
            if sql_query:
                with st.expander("Generated SQL Query"):
                    st.code(sql_query, language="sql")
                st.info("🔍 Executing query on BigQuery...")
                df, error = st.session_state.query_engine.execute_sql_query(sql_query)
                if error:
                    st.error(f"Query Error: {error}")
                else:
                    response = st.session_state.query_engine.format_results_to_natural_language(df, user_query)
                    st.success("✅ Query executed successfully!")
                    st.markdown("### 📊 Answer:")
                    st.markdown(f"**{response}**")
                    if st.checkbox("Show raw data"):
                        st.dataframe(df)
            else:
                st.error("Sorry, I couldn't generate a valid SQL query. Please try rephrasing.")
    
    st.markdown("---")
    st.markdown("*Powered by OpenAI GPT-3.5 and Google BigQuery*")

if __name__ == "__main__":
    main()
