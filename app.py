import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import BadRequest, NotFound
from openai import OpenAI
import re
import logging
from prophet import Prophet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="NL to SQL Analytics & Recommendations", page_icon="🔍", layout="wide")

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

class NL2AnalyticsEngine:
    def __init__(self):
        self.project_id = st.secrets["project_id"]
        self.dataset_id = "locatify_dataset_1"
        self.table_id = "brand_orders"
        self.bq = None
        self.oa = None
        self.dataset_location = None
        self.schema = self._get_table_schema()
        self._setup_clients()
        self._verify_dataset_access()

    def _get_table_schema(self):
        """Restituisce lo schema della tabella con i tipi di dato"""
        return {
            "unique_id": "STRING",
            "order_id": "STRING", 
            "channel": "STRING",
            "order_date_timestamp": "TIMESTAMP",
            "full_name": "STRING",
            "street": "STRING", 
            "house_no": "STRING",
            "postal_code": "STRING",
            "city": "STRING",
            "country_code": "STRING",
            "email": "STRING",
            "total_eur": "FLOAT",
            "customer_id": "STRING",
            "latitude": "FLOAT",
            "longitude": "FLOAT",
            "inserted_at": "TIMESTAMP"
        }
    
    def _setup_clients(self):
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        # Inizializza il client BigQuery
        self.bq = bigquery.Client(project=self.project_id, credentials=creds)
        self.oa = OpenAI(api_key=st.secrets["openai_api_key"])

    def _verify_dataset_access(self):
        """Verifica l'accesso al dataset e rileva la location"""
        try:
            # Prima prova a listare i dataset disponibili
            datasets = list(self.bq.list_datasets())
            available_datasets = [d.dataset_id for d in datasets]
            
            if self.dataset_id not in available_datasets:
                st.error(f"Dataset '{self.dataset_id}' non trovato!")
                st.info(f"Dataset disponibili: {available_datasets}")
                return False
            
            # Ottieni informazioni sul dataset
            dataset_ref = self.bq.dataset(self.dataset_id)
            dataset = self.bq.get_dataset(dataset_ref)
            self.dataset_location = dataset.location
            
            # Verifica che la tabella esista
            try:
                table_ref = dataset_ref.table(self.table_id)
                table = self.bq.get_table(table_ref)
                st.success(f"✅ Dataset e tabella trovati! Location: {self.dataset_location}")
                return True
            except NotFound:
                # Lista le tabelle disponibili
                tables = list(self.bq.list_tables(dataset))
                available_tables = [t.table_id for t in tables]
                st.error(f"Tabella '{self.table_id}' non trovata!")
                st.info(f"Tabelle disponibili: {available_tables}")
                return False
                
        except Exception as e:
            st.error(f"Errore nella verifica del dataset: {e}")
            return False

    @property
    def full_table_name(self):
        return f"`{self.project_id}.{self.dataset_id}.{self.table_id}`"

    def classify_request(self, nl: str) -> str:
        system_prompt = (
            "You are an analytics assistant. "
            "Classify the user's request into exactly one of: "
            "top_n, distribution, correlation, forecast, raw, summary, recommendation, investment_analysis."
        )
        response = self.oa.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Request: \"{nl}\""}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()

    def generate_sql(self, nl: str) -> str:
        # Crea la descrizione delle colonne per il prompt
        columns_info = []
        for col_name, col_type in self.schema.items():
            columns_info.append(f"- {col_name} ({col_type})")
        
        columns_desc = "\n".join(columns_info)
        
        prompt = (
            f"You are a BigQuery SQL expert. Given the request:\n\"{nl}\"\n\n"
            f"Generate only the SQL (no explanations) on table {self.full_table_name}.\n\n"
            f"IMPORTANT: Use EXACTLY these column names:\n{columns_desc}\n\n"
            f"Common mappings:\n"
            f"- For dates/time: use 'order_date_timestamp' (TIMESTAMP)\n"
            f"- For order value/amount/price: use 'total_eur' (FLOAT)\n"
            f"- For customer/user: use 'customer_id' (STRING)\n"
            f"- For sales channel: use 'channel' (STRING)\n"
            f"- For location: use 'city' and 'country_code'\n\n"
            f"When working with timestamps, use DATE() function to extract date portion if needed."
        )
        
        response = self.oa.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "BigQuery SQL expert with exact schema knowledge"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        sql = response.choices[0].message.content
        return re.sub(r'```.*?\n|```', '', sql).strip()

    def execute_query(self, sql: str):
        """Esegue la query con gestione errori migliorata"""
        try:
            # Esegui la query direttamente senza specificare location nel job config
            job = self.bq.query(sql)
            return job.result().to_dataframe()
            
        except BadRequest as e:
            msg = e.message or str(e)
            
            # Gestione errori di colonne non riconosciute
            m = re.search(r"Unrecognized name: (\w+)", msg, re.IGNORECASE)
            if m:
                bad_col = m.group(1)
                col_map = {
                    "order_date": "order_date_timestamp",
                    "date": "order_date_timestamp", 
                    "order_time": "order_date_timestamp",
                    "timestamp": "order_date_timestamp",
                    "total_amount": "total_eur",
                    "amount": "total_eur",
                    "spend": "total_eur",
                    "price": "total_eur",
                    "revenue": "total_eur",
                    "sales": "total_eur",
                    "user_id": "customer_id",
                    "customer": "customer_id"
                }
                if bad_col in col_map:
                    corrected = col_map[bad_col]
                    st.warning(f"Colonna `{bad_col}` non trovata; sostituisco con `{corrected}`")
                    sql_fixed = re.sub(rf"\b{bad_col}\b", corrected, sql, flags=re.IGNORECASE)
                    st.markdown("### SQL Corretto")
                    st.code(sql_fixed, language="sql")
                    
                    # Riprova con SQL corretto
                    try:
                        job = self.bq.query(sql_fixed)
                        return job.result().to_dataframe()
                    except Exception as e2:
                        st.error(f"Errore con SQL corretto: {e2}")
                        return None
                else:
                    st.error(f"Colonna `{bad_col}` non riconosciuta e non mappata")
                    # Mostra le colonne disponibili
                    available_cols = list(self.schema.keys())
                    st.info(f"Colonne disponibili: {', '.join(available_cols)}")
                    return None
            else:
                st.error(f"Errore SQL: {msg}")
                return None
                
        except NotFound as e:
            st.error(f"Dataset/tabella non trovati: {e}")
            return None
        except Exception as e:
            st.error(f"Errore esecuzione query: {e}")
            return None

    def run(self, user_query: str):
        # Debug command
        if user_query.lower() == "debug":
            self._verify_dataset_access()
            return
            
        q_lower = user_query.lower()
        if 'invest' in q_lower or 'why' in q_lower:
            analysis = 'recommendation'
        else:
            analysis = self.classify_request(user_query)

        sql = self.generate_sql(user_query)
        st.markdown("### Generated SQL Query")
        st.code(sql, language="sql")

        df = self.execute_query(sql)
        if df is None or df.empty:
            st.warning("Nessun risultato ottenuto.")
            return

        # Resto del codice per l'analisi...
        if analysis == "recommendation":
            metrics_sql = f"""
            SELECT city,
                   COUNT(order_id) AS order_count,
                   SUM(total_eur) AS total_revenue,
                   AVG(total_eur) AS avg_order_value
            FROM {self.full_table_name}
            GROUP BY city
            ORDER BY total_revenue DESC
            LIMIT 10;"""
            
            metrics_df = self.execute_query(metrics_sql)
            if metrics_df is not None and not metrics_df.empty:
                st.markdown("### City Metrics")
                st.dataframe(metrics_df)
                
                csv = metrics_df.to_csv(index=False)
                advice_prompt = (
                    "You are a data-driven investment advisor. "
                    "Given the following city metrics (orders, revenue, avg order value), "
                    "recommend which cities to invest in and why. Use the data to justify your answer.\n\n"
                    f"```csv\n{csv}\n```"
                )
                advice = self.oa.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"Financial advisor"},
                        {"role":"user","content":advice_prompt}
                    ],
                    temperature=0.7
                ).choices[0].message.content
                st.markdown("### Investment Recommendation")
                st.markdown(advice)
            return

        # Altri tipi di analisi...
        if analysis == "top_n":
            if len(df.columns) >= 2:
                c1, c2 = df.columns[:2]
                lines = [f"Top results by {c2}:"]
                for idx, row in df.iterrows():
                    if idx >= 10:  # Limita a 10 risultati
                        break
                    lines.append(f"{idx+1}. {row[c1]} → {row[c2]:,.0f}")
                st.markdown("\n".join(lines))
            else:
                st.dataframe(df)

        elif analysis == "distribution":
            col = df.columns[0]
            counts = df[col].value_counts().head(10)
            st.bar_chart(counts)
            st.markdown(f"Distribution of **{col}** (top 10)")

        elif analysis == "correlation":
            # Seleziona solo colonne numeriche
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                st.markdown("### Correlation Matrix")
                st.dataframe(corr)
            else:
                st.warning("Non ci sono abbastanza colonne numeriche per calcolare la correlazione")

        elif analysis == "forecast":
            if len(df.columns) >= 2:
                try:
                    ds = pd.to_datetime(df.iloc[:, 0])
                    y = df.iloc[:, 1]
                    df_ts = pd.DataFrame({"ds": ds, "y": y}).sort_values("ds")
                    
                    model = Prophet()
                    model.fit(df_ts)
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)
                    
                    st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])  
                    st.markdown("### Forecast for next 30 periods")
                    
                    # Mostra anche i dati storici
                    st.line_chart(df_ts.set_index("ds"))
                    st.markdown("### Historical Data")
                except Exception as e:
                    st.error(f"Errore nel forecasting: {e}")
                    st.markdown("### Raw Data")
                    st.dataframe(df)
            else:
                st.warning("Servono almeno 2 colonne per il forecasting")

        elif analysis == "raw":
            st.markdown("### Raw Data")
            st.dataframe(df)

        else:  # summary
            st.markdown("### Summary (top 10 rows)")
            st.dataframe(df.head(10))
            st.markdown(f"Found **{len(df)}** rows total.")


def main():
    st.title("🔍 Advanced Analytics & Investment Advisor")
    st.markdown("*Powered by AI-driven data analysis and predictive insights*")
    
    # Mostra info sulla configurazione
    with st.expander("ℹ️ Available Analysis Types"):
        st.markdown("""
        **📊 Standard Analytics:**
        - Top N results, distributions, correlations
        - Time series forecasting with Prophet
        - Raw data exploration and summaries
        
        **🎯 Investment Analysis:**
        - Comprehensive market analysis by city and channel
        - Customer segmentation and behavior patterns
        - Seasonal trends and growth predictions
        - AI-powered investment recommendations
        
        **💡 Example Questions:**
        - "Where should I invest more?" → Full investment analysis
        - "Show me sales trends by month" → Time series analysis
        - "Which cities perform best?" → City comparison
        - "Forecast next month's orders" → Predictive modeling
        """)
    
    user_query = st.text_input("Ask anything about your business data...", placeholder="e.g., Where should I invest more?")
    
    if st.button("🚀 Analyze", type="primary"):
        if user_query:
            st.session_state.query_history.append(user_query)
            with st.spinner("🔄 Analyzing your data and generating insights..."):
                engine = NL2AnalyticsEngine()
                engine.run(user_query)
        else:
            st.warning("Please enter a question!")

    # Sidebar con cronologia e suggerimenti
    if st.session_state.query_history:
        st.sidebar.header("📝 Recent Queries")
        for q in st.session_state.query_history[-5:]:
            if st.sidebar.button(f"🔄 {q[:30]}...", key=f"rerun_{q}"):
                engine = NL2AnalyticsEngine()
                engine.run(q)
    
    # Suggerimenti nella sidebar
    st.sidebar.header("💡 Suggested Analyses")
    suggestions = [
        "Where should I invest more?",
        "Show me monthly sales trends",
        "Which channels perform best?",
        "Forecast next month orders",
        "Customer segmentation analysis"
    ]
    
    for suggestion in suggestions:
        if st.sidebar.button(f"📊 {suggestion}", key=f"suggest_{suggestion}"):
            engine = NL2AnalyticsEngine()
            engine.run(suggestion)

if __name__ == "__main__":
    main()