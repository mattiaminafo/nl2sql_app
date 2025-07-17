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
            "top_n, distribution, correlation, forecast, raw, summary, recommendation.\n\n"
            "IMPORTANT: Use 'forecast' for any request about predicting future values, "
            "trends, or 'next month/week/year' data, even if the word 'forecast' is not used."
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

    def generate_forecast_sql(self, nl: str) -> str:
        """Genera SQL specifico per raccogliere dati storici per forecasting"""
        prompt = (
            f"The user wants to forecast: \"{nl}\"\n\n"
            f"Generate SQL to get HISTORICAL TIME SERIES DATA from {self.full_table_name}.\n\n"
            f"⚠️ CRITICAL: NEVER use future dates or WHERE conditions that look for future data!\n"
            f"📊 ALWAYS get historical data to train a forecasting model\n\n"
            f"RULES:\n"
            f"- Get ALL historical data available\n"
            f"- Use DATE(order_date_timestamp) for daily aggregation\n"
            f"- ALWAYS ORDER BY date ASC for time series\n"
            f"- If user mentions 'by city' or location, group by both date AND city\n\n"
            f"EXAMPLES:\n"
            f"📈 'forecast next month orders' → \n"
            f"SELECT DATE(order_date_timestamp) as date, COUNT(*) as orders \n"
            f"FROM {self.full_table_name} \n"
            f"GROUP BY DATE(order_date_timestamp) \n"
            f"ORDER BY date ASC\n\n"
            f"🏙️ 'forecast next month orders by city' → \n"
            f"SELECT DATE(order_date_timestamp) as date, city, COUNT(*) as orders \n"
            f"FROM {self.full_table_name} \n"
            f"GROUP BY DATE(order_date_timestamp), city \n"
            f"ORDER BY date ASC, city\n\n"
            f"💰 'forecast revenue by channel' → \n"
            f"SELECT DATE(order_date_timestamp) as date, channel, SUM(total_eur) as revenue \n"
            f"FROM {self.full_table_name} \n"
            f"GROUP BY DATE(order_date_timestamp), channel \n"
            f"ORDER BY date ASC, channel\n\n"
            f"Available columns: {', '.join(self.schema.keys())}\n\n"
            f"🎯 Generate ONLY the SQL for HISTORICAL data, no explanations."
        )
        
        response = self.oa.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "SQL expert: Generate ONLY historical data queries for forecasting, NEVER future data"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        sql = response.choices[0].message.content
        return re.sub(r'```.*?\n|```', '', sql).strip()
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
            f"BIGQUERY TIMESTAMP RULES:\n"
            f"- Use DATE(order_date_timestamp) to extract date from timestamp\n"
            f"- Use DATETIME(order_date_timestamp) to extract datetime from timestamp\n"
            f"- For date arithmetic, convert to DATE first: DATE_ADD(DATE(order_date_timestamp), INTERVAL 1 MONTH)\n"
            f"- Use DATE_SUB() and DATE_ADD() for date operations, NOT TIMESTAMP_ADD with MONTH\n"
            f"- For time-based filtering: WHERE order_date_timestamp >= '2024-01-01' AND order_date_timestamp < '2024-02-01'\n"
            f"- For grouping by date: GROUP BY DATE(order_date_timestamp)\n"
            f"- For extracting year/month: EXTRACT(YEAR FROM order_date_timestamp), EXTRACT(MONTH FROM order_date_timestamp)\n\n"
            f"Examples:\n"
            f"- Daily trends: SELECT DATE(order_date_timestamp) as order_date, COUNT(*) FROM table GROUP BY DATE(order_date_timestamp)\n"
            f"- Monthly trends: SELECT DATE_TRUNC(DATE(order_date_timestamp), MONTH) as month, COUNT(*) FROM table GROUP BY month\n"
            f"- Last 30 days: WHERE order_date_timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)\n"
            f"- Next month data: WHERE DATE(order_date_timestamp) >= DATE_ADD(CURRENT_DATE(), INTERVAL 1 MONTH)"
        )
        
        response = self.oa.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "BigQuery SQL expert with exact schema and timestamp handling knowledge"},
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
            
            # Gestione errori specifici di BigQuery TIMESTAMP
            elif "TIMESTAMP_ADD does not support the MONTH date part" in msg:
                st.warning("Errore TIMESTAMP_ADD con MONTH - correggo con DATE_ADD")
                # Sostituisci TIMESTAMP_ADD con DATE_ADD e converti a DATE
                sql_fixed = re.sub(
                    r'TIMESTAMP_ADD\s*\(\s*([^,]+),\s*INTERVAL\s+(\d+)\s+MONTH\s*\)',
                    r'DATE_ADD(DATE(\1), INTERVAL \2 MONTH)',
                    sql,
                    flags=re.IGNORECASE
                )
                st.markdown("### SQL Corretto")
                st.code(sql_fixed, language="sql")
                
                try:
                    job = self.bq.query(sql_fixed)
                    return job.result().to_dataframe()
                except Exception as e2:
                    st.error(f"Errore con SQL corretto: {e2}")
                    return None
                    
            elif "does not support the MONTH date part when the argument is TIMESTAMP type" in msg:
                st.warning("Errore operazione MONTH su TIMESTAMP - correggo convertendo a DATE")
                # Trova e correggi operazioni di data su timestamp
                sql_fixed = re.sub(
                    r'([A-Z_]+)\s*\(\s*(order_date_timestamp),\s*INTERVAL\s+(\d+)\s+MONTH\s*\)',
                    r'\1(DATE(\2), INTERVAL \3 MONTH)',
                    sql,
                    flags=re.IGNORECASE
                )
                st.markdown("### SQL Corretto")
                st.code(sql_fixed, language="sql")
                
                try:
                    job = self.bq.query(sql_fixed)
                    return job.result().to_dataframe()
                except Exception as e2:
                    st.error(f"Errore con SQL corretto: {e2}")
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
                    # Gestisci forecast multi-dimensionale (per città, canale, etc.)
                    if len(df.columns) == 3:  # data, dimensione, valore
                        st.markdown("### Multi-dimensional Forecast")
                        
                        # Ordina per data
                        df_sorted = df.sort_values(df.columns[0])
                        
                        # Ottieni le dimensioni uniche (es. città)
                        dimension_col = df.columns[1]
                        value_col = df.columns[2]
                        dimensions = df_sorted[dimension_col].unique()
                        
                        # Limita a top 5 dimensioni per performance
                        if len(dimensions) > 5:
                            top_dimensions = df_sorted.groupby(dimension_col)[value_col].sum().nlargest(5).index
                            df_sorted = df_sorted[df_sorted[dimension_col].isin(top_dimensions)]
                            dimensions = top_dimensions
                        
                        st.markdown(f"### Historical Data by {dimension_col}")
                        
                        # Crea un chart per dimensione
                        forecasts = {}
                        for dim in dimensions:
                            dim_data = df_sorted[df_sorted[dimension_col] == dim]
                            if len(dim_data) >= 10:  # Minimo dati per forecast
                                ds = pd.to_datetime(dim_data.iloc[:, 0])
                                y = dim_data.iloc[:, 2].astype(float)
                                df_ts = pd.DataFrame({"ds": ds, "y": y})
                                
                                # Mostra dati storici
                                st.write(f"**{dim}** - Historical trend:")
                                st.line_chart(df_ts.set_index("ds"))
                                
                                # Forecast
                                try:
                                    model = Prophet(
                                        daily_seasonality=True,
                                        weekly_seasonality=True,
                                        yearly_seasonality=False
                                    )
                                    model.fit(df_ts)
                                    
                                    # Determina periodo
                                    if 'month' in user_query.lower():
                                        forecast_periods = 30
                                    elif 'week' in user_query.lower():
                                        forecast_periods = 7
                                    else:
                                        forecast_periods = 30
                                    
                                    future = model.make_future_dataframe(periods=forecast_periods)
                                    forecast = model.predict(future)
                                    
                                    # Salva forecast per summary
                                    future_only = forecast.tail(forecast_periods)
                                    forecasts[dim] = future_only["yhat"].sum()
                                    
                                    # Mostra forecast
                                    st.write(f"**{dim}** - Forecast:")
                                    st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
                                    
                                except Exception as e:
                                    st.warning(f"Errore nel forecast per {dim}: {e}")
                        
                        # Summary finale
                        if forecasts:
                            st.markdown("### 📊 Forecast Summary")
                            summary_df = pd.DataFrame(list(forecasts.items()), 
                                                    columns=[dimension_col, "Predicted_Total"])
                            summary_df = summary_df.sort_values("Predicted_Total", ascending=False)
                            st.dataframe(summary_df)
                            
                            total_all = summary_df["Predicted_Total"].sum()
                            st.metric("Total Predicted", f"{total_all:,.0f}")
                    
                    else:
                        # Forecast semplice (data, valore)
                        df_sorted = df.sort_values(df.columns[0])
                        
                        # Converte la prima colonna in datetime
                        ds = pd.to_datetime(df_sorted.iloc[:, 0])
                        y = df_sorted.iloc[:, 1].astype(float)
                        
                        # Crea DataFrame per Prophet
                        df_ts = pd.DataFrame({"ds": ds, "y": y})
                        
                        st.markdown("### Historical Data")
                        st.line_chart(df_ts.set_index("ds"))
                        
                        # Verifica che ci siano abbastanza dati
                        if len(df_ts) < 10:
                            st.warning("Non ci sono abbastanza dati storici per un forecast accurato")
                            st.dataframe(df_ts)
                            return
                        
                        # Addestra il modello Prophet
                        model = Prophet(
                            daily_seasonality=True,
                            weekly_seasonality=True,
                            yearly_seasonality=True if len(df_ts) > 365 else False
                        )
                        model.fit(df_ts)
                        
                        # Determina il periodo di forecast basato sulla query
                        if 'month' in user_query.lower():
                            forecast_periods = 30
                            period_name = "30 days (next month)"
                        elif 'week' in user_query.lower():
                            forecast_periods = 7
                            period_name = "7 days (next week)"
                        elif 'year' in user_query.lower():
                            forecast_periods = 365
                            period_name = "365 days (next year)"
                        else:
                            forecast_periods = 30
                            period_name = "30 days"
                        
                        # Genera forecast
                        future = model.make_future_dataframe(periods=forecast_periods)
                        forecast = model.predict(future)
                        
                        # Mostra forecast
                        st.markdown(f"### Forecast for {period_name}")
                        forecast_chart = forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
                        st.line_chart(forecast_chart)
                        
                        # Mostra le predizioni future
                        future_forecast = forecast.tail(forecast_periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                        future_forecast.columns = ["Date", "Predicted", "Lower_Bound", "Upper_Bound"]
                        st.markdown("### Future Predictions")
                        st.dataframe(future_forecast)
                        
                        # Calcola totale previsto
                        total_predicted = future_forecast["Predicted"].sum()
                        st.markdown(f"### Summary")
                        st.metric("Total Predicted Orders", f"{total_predicted:,.0f}")
                        
                except Exception as e:
                    st.error(f"Errore nel forecasting: {e}")
                    st.markdown("### Raw Data")
                    st.dataframe(df)
            else:
                st.warning("Servono almeno 2 colonne per il forecasting")
                st.dataframe(df)

        elif analysis == "raw":
            st.markdown("### Raw Data")
            st.dataframe(df)

        else:  # summary
            st.markdown("### Summary (top 10 rows)")
            st.dataframe(df.head(10))
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