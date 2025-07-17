import streamlit as st
import pandas as pd
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
        self.dataset_id = "locatify_data"
        self.table_id = "brand_orders"
        self.bq = None
        self.oa = None
        self.dataset_location = None
        self._setup_clients()
        self._verify_dataset_access()

    def _setup_clients(self):
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        # Inizializza senza specificare location
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
            "top_n, distribution, correlation, forecast, raw, summary, recommendation."
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
        prompt = (
            f"You are a BigQuery SQL expert. Given the request:\n\"{nl}\"\n"
            f"Generate only the SQL (no explanations) on table {self.full_table_name}.\n"
            f"Available columns likely include: order_id, channel, order_date, city, country_code, total_eur, customer_id"
        )
        response = self.oa.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "SQL expert"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        sql = response.choices[0].message.content
        return re.sub(r'```.*?\n|```', '', sql).strip()

    def execute_query(self, sql: str):
        """Esegue la query con gestione errori migliorata"""
        try:
            # Configura il job con la location corretta se disponibile
            job_config = bigquery.QueryJobConfig()
            if self.dataset_location:
                job_config.location = self.dataset_location
            
            job = self.bq.query(sql, job_config=job_config)
            return job.result().to_dataframe()
            
        except BadRequest as e:
            msg = e.message or str(e)
            
            # Gestione errori di colonne non riconosciute
            m = re.search(r"Unrecognized name: (\w+)", msg, re.IGNORECASE)
            if m:
                bad_col = m.group(1)
                col_map = {
                    "total_amount": "total_eur",
                    "amount": "total_eur",
                    "spend": "total_eur",
                    "price": "total_eur",
                    "user_id": "customer_id"
                }
                if bad_col in col_map:
                    corrected = col_map[bad_col]
                    st.warning(f"Colonna `{bad_col}` non trovata; sostituisco con `{corrected}`")
                    sql_fixed = re.sub(rf"\b{bad_col}\b", corrected, sql, flags=re.IGNORECASE)
                    st.markdown("### SQL Corretto")
                    st.code(sql_fixed, language="sql")
                    
                    # Riprova con SQL corretto
                    try:
                        job = self.bq.query(sql_fixed, job_config=job_config)
                        return job.result().to_dataframe()
                    except Exception as e2:
                        st.error(f"Errore con SQL corretto: {e2}")
                        return None
                else:
                    st.error(f"Colonna `{bad_col}` non riconosciuta e non mappata")
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
            c1, c2 = df.columns[:2]
            lines = [f"Top results by {c2}:"]
            for idx, row in df.iterrows():
                lines.append(f"{idx+1}. {row[c1]} → {row[c2]:,.0f}")
            st.markdown("\n".join(lines))

        elif analysis == "distribution":
            col = df.columns[0]
            counts = df[col].value_counts().head(10)
            st.bar_chart(counts)
            st.markdown(f"Distribution of **{col}** (top 10)")

        elif analysis == "correlation":
            corr = df.corr()
            st.markdown("### Correlation Matrix")
            st.dataframe(corr)

        elif analysis == "forecast":
            ds = pd.to_datetime(df.iloc[:, 0])
            y = df.iloc[:, 1]
            df_ts = pd.DataFrame({"ds": ds, "y": y})
            model = Prophet()
            model.fit(df_ts)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])  
            st.markdown("Forecast for next 30 periods")

        elif analysis == "raw":
            st.markdown("### Raw Data")
            st.dataframe(df)

        else:  # summary
            st.markdown("### Summary (top 5 rows)")
            st.table(df.head(5))
            st.markdown(f"Found **{len(df)}** rows total.")


def main():
    st.title("🔍 NL Analytics & Investment Advisor")
    
    # Mostra info sulla configurazione
    with st.expander("ℹ️ Configuration Info"):
        st.write("Per debug, scrivi 'debug' e premi Run")
    
    user_query = st.text_input("Ask anything (analytics/statistics/forecast/invest)...")
    if st.button("Run"):
        if user_query:
            st.session_state.query_history.append(user_query)
            with st.spinner("Processing your request…"):
                engine = NL2AnalyticsEngine()
                engine.run(user_query)
        else:
            st.warning("Inserisci una domanda!")

    if st.session_state.query_history:
        st.sidebar.header("Recent Queries")
        for q in st.session_state.query_history[-5:]:
            st.sidebar.write(f"- {q}")

if __name__ == "__main__":
    main()