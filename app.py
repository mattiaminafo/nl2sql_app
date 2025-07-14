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
            # Setup BigQuery client
            if "gcp_service_account" in st.secrets:
                # Use service account from secrets
                credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                self.bq_client = bigquery.Client(credentials=credentials)
            elif "project_id" in st.secrets:
                # Use project ID with default credentials
                self.bq_client = bigquery.Client(project=st.secrets["project_id"])
            else:
                # Try with default credentials (for local development)
                try:
                    self.bq_client = bigquery.Client(project="planar-flux-465609-e1")
                except Exception:
                    st.error("BigQuery credentials not found. Please configure authentication.")
                    return
            
            # Setup OpenAI client
            if "openai_api_key" in st.secrets:
                self.openai_client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
            else:
                st.error("OpenAI API key not found in secrets")
                return
                
        except Exception as e:
            logger.error(f"Error setting up clients: {str(e)}")
            st.error(f"Error setting up clients: {str(e)}")
    
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
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a SQL expert that converts natural language to BigQuery SQL."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            st.error(f"OpenAI API Error: {str(e)}")
            return None
    
    def execute_sql_query(self, sql_query):
        """Execute SQL query on BigQuery"""
        try:
            # Validate query contains our table
            if self.dataset_name not in sql_query:
                return None, "Query must reference the correct dataset table"
            
            # Execute query
            query_job = self.bq_client.query(sql_query)
            results = query_job.result()
            
            # Convert to DataFrame
            df = results.to_dataframe()
            
            if df.empty:
                return df, "No results found for your query"
            
            return df, None
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return None, f"Error executing query: {str(e)}"
    
    def format_results_to_natural_language(self, df, original_query):
        """Convert query results to natural language response"""
        if df is None or df.empty:
            return "No results found for your query."
        
        try:
            # Handle single value results
            if len(df.columns) == 1 and len(df) == 1:
                col_name = df.columns[0]
                value = df.iloc[0, 0]
                
                if "count" in col_name.lower() or "total" in col_name.lower():
                    return f"The result is {value:,.0f}"
                elif "average" in col_name.lower() or "avg" in col_name.lower():
                    return f"The average value is {value:,.2f}"
                else:
                    return f"The result is {value}"
            
            # Handle top results (city, country, channel, etc.)
            elif len(df.columns) == 2 and len(df) >= 1:
                col1, col2 = df.columns
                top_result = df.iloc[0]
                
                if "city" in original_query.lower():
                    return f"The city with the most orders is {top_result[col1]} with {top_result[col2]:,.0f} orders"
                elif "country" in original_query.lower():
                    return f"The country with the highest value is {top_result[col1]} with {top_result[col2]:,.2f}"
                elif "channel" in original_query.lower():
                    return f"The channel with the most orders is {top_result[col1]} with {top_result[col2]:,.0f} orders"
                else:
                    return f"The top result is {top_result[col1]} with {top_result[col2]}"
            
            # Handle multiple results
            else:
                if len(df) <= 5:
                    result_text = "Here are the results:\n"
                    for _, row in df.iterrows():
                        result_text += f"• {', '.join([f'{col}: {val}' for col, val in row.items()])}\n"
                    return result_text
                else:
                    return f"Found {len(df)} results. Here are the top 5:\n" + \
                           df.head().to_string(index=False)
                           
        except Exception as e:
            logger.error(f"Error formatting results: {str(e)}")
            return f"Results found but formatting error: {str(e)}"

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
        Try asking questions like:
        - Which city has the most orders?
        - Which channel does the most orders in July?
        - What is the average order value?
        - How many orders were placed last month?
        - Which country has the highest total revenue?
        - Show me the top 5 cities by order count
        """)
        
        if st.session_state.query_history:
            st.header("🕐 Recent Queries")
            for i, query in enumerate(st.session_state.query_history[-5:]):
                st.text(f"{i+1}. {query[:50]}...")
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_input(
            "Enter your question in English:",
            placeholder="e.g., Which city has the most orders?",
            help="Ask any question about your orders data"
        )
    
    with col2:
        ask_button = st.button("Ask Question", type="primary")
    
    # Process query
    if ask_button and user_query:
        with st.spinner("Processing your question..."):
            try:
                # Add to history
                st.session_state.query_history.append(user_query)
                
                # Generate SQL
                st.info("🤖 Converting your question to SQL...")
                sql_query = st.session_state.query_engine.generate_sql_from_nl(user_query)
                
                if sql_query:
                    # Show generated SQL (optional)
                    with st.expander("Generated SQL Query"):
                        st.code(sql_query, language="sql")
                    
                    # Execute query
                    st.info("🔍 Executing query on BigQuery...")
                    df, error = st.session_state.query_engine.execute_sql_query(sql_query)
                    
                    if error:
                        st.error(f"Query Error: {error}")
                    else:
                        # Format response
                        response = st.session_state.query_engine.format_results_to_natural_language(df, user_query)
                        
                        # Display results
                        st.success("✅ Query executed successfully!")
                        st.markdown("### 📊 Answer:")
                        st.markdown(f"**{response}**")
                        
                        # Show raw data if requested
                        if st.checkbox("Show raw data"):
                            st.dataframe(df)
                
                else:
                    st.error("Sorry, I couldn't generate a valid SQL query from your question. Please try rephrasing.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Main error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by OpenAI GPT-3.5 and Google BigQuery*")

if __name__ == "__main__":
    main()