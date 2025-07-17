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
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Smart Analytics Assistant", page_icon="🤖", layout="wide")

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

class SmartAnalyticsEngine:
    def __init__(self):
        self.project_id = st.secrets["project_id"]
        self.dataset_id = "locatify_dataset_1"
        self.table_id = "brand_orders"
        self.full_table_name = f"`{self.project_id}.{self.dataset_id}.{self.table_id}`"
        
        # Schema del database
        self.schema = {
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
        
        self.bq = None
        self.oa = None
        self._setup_clients()

    def _setup_clients(self):
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        self.bq = bigquery.Client(project=self.project_id, credentials=creds)
        self.oa = OpenAI(api_key=st.secrets["openai_api_key"])

    def analyze_question(self, question: str) -> dict:
        """Analizza la domanda e determina il tipo di risposta necessaria"""
        
        # Keywords per identificare il tipo di analisi
        forecasting_keywords = [
            'previsione', 'prevision', 'forecast', 'predict', 'futuro', 'future',
            'prossimo', 'next', 'dovrei investire', 'should invest', 'raccomanda',
            'recommend', 'strategia', 'strategy', 'migliore', 'best', 'quale città',
            'which city', 'dove', 'where', 'perché', 'why', 'motivazione'
        ]
        
        time_keywords = [
            'trend', 'andamento', 'nel tempo', 'over time', 'crescita', 'growth',
            'stagionalità', 'seasonal', 'mensile', 'monthly', 'giornaliero', 'daily'
        ]
        
        q_lower = question.lower()
        
        # Determina il tipo di analisi
        is_forecasting = any(keyword in q_lower for keyword in forecasting_keywords)
        is_time_series = any(keyword in q_lower for keyword in time_keywords) or is_forecasting
        
        return {
            'is_forecasting': is_forecasting,
            'is_time_series': is_time_series,
            'original_question': question
        }

    def generate_sql(self, question: str, analysis_type: dict) -> str:
        """Genera SQL intelligente basato sulla domanda"""
        
        # Crea descrizione dello schema
        schema_desc = "\n".join([f"- {col}: {dtype}" for col, dtype in self.schema.items()])
        
        # Context specifico per il tipo di analisi
        if analysis_type['is_forecasting']:
            context = (
                "L'utente vuole previsioni o raccomandazioni strategiche. "
                "Genera SQL per ottenere dati storici che permettano analisi predittive. "
                "Includi sempre una dimensione temporale (DATE(order_date_timestamp)) quando possibile."
            )
        elif analysis_type['is_time_series']:
            context = (
                "L'utente vuole analisi temporali o trend. "
                "Genera SQL che includa DATE(order_date_timestamp) per analisi nel tempo."
            )
        else:
            context = (
                "L'utente vuole un'analisi descrittiva o statistica. "
                "Genera SQL per rispondere direttamente alla domanda."
            )

        prompt = f"""
Sei un esperto di BigQuery SQL. L'utente ha fatto questa domanda: "{question}"

CONTESTO: {context}

SCHEMA TABELLA {self.full_table_name}:
{schema_desc}

REGOLE IMPORTANTI:
- Usa ESATTAMENTE i nomi delle colonne dello schema
- Per date usa: DATE(order_date_timestamp)
- Per ordini usa: COUNT(*) o COUNT(order_id) 
- Per revenue usa: SUM(total_eur)
- Per clienti usa: customer_id
- Sempre ORDER BY appropriato
- Se è un'analisi temporale, includi sempre la data

ESEMPI:
- "quanti ordini questo mese" → WHERE DATE(order_date_timestamp) >= DATE_TRUNC(CURRENT_DATE(), MONTH)
- "cliente che spende di più" → GROUP BY customer_id ORDER BY SUM(total_eur) DESC
- "trend mensile ordini" → GROUP BY DATE_TRUNC(DATE(order_date_timestamp), MONTH)

Genera SOLO il SQL, senza spiegazioni:
"""

        response = self.oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Esperto SQL che genera query precise e ottimizzate"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        sql = response.choices[0].message.content
        return re.sub(r'```.*?\n|```', '', sql).strip()

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Esegue SQL con gestione errori intelligente"""
        try:
            job = self.bq.query(sql)
            return job.result().to_dataframe()
        except BadRequest as e:
            msg = str(e)
            
            # Auto-correzione errori comuni
            if "Unrecognized name" in msg:
                # Estrai il nome della colonna errata
                match = re.search(r"Unrecognized name: (\w+)", msg)
                if match:
                    wrong_col = match.group(1)
                    
                    # Mappature comuni
                    col_fixes = {
                        'order_date': 'order_date_timestamp',
                        'date': 'order_date_timestamp',
                        'amount': 'total_eur',
                        'total_amount': 'total_eur',
                        'revenue': 'total_eur',
                        'user_id': 'customer_id'
                    }
                    
                    if wrong_col in col_fixes:
                        fixed_sql = re.sub(rf'\b{wrong_col}\b', col_fixes[wrong_col], sql)
                        st.warning(f"🔧 Corretto: {wrong_col} → {col_fixes[wrong_col]}")
                        st.code(fixed_sql, language="sql")
                        return self.execute_sql(fixed_sql)
            
            st.error(f"❌ Errore SQL: {msg}")
            return None
        except Exception as e:
            st.error(f"❌ Errore esecuzione: {e}")
            return None

    def create_visualization(self, df: pd.DataFrame, question: str) -> None:
        """Crea visualizzazioni intelligenti basate sui dati"""
        if df is None or df.empty:
            return
            
        # Identifica colonne temporali
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Se ci sono date, crea grafici temporali
        if date_cols and numeric_cols:
            date_col = date_cols[0]
            if len(numeric_cols) > 0:
                df[date_col] = pd.to_datetime(df[date_col])
                df_sorted = df.sort_values(date_col)
                
                for num_col in numeric_cols[:2]:  # Max 2 metriche
                    fig = px.line(df_sorted, x=date_col, y=num_col, 
                                title=f"{num_col.title()} nel tempo")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Grafici per top rankings
        elif len(df.columns) >= 2 and len(df) <= 20:
            if df.dtypes.iloc[1] in ['int64', 'float64']:
                fig = px.bar(df.head(10), x=df.columns[0], y=df.columns[1],
                           title=f"Top {df.columns[0]} per {df.columns[1]}")
                st.plotly_chart(fig, use_container_width=True)

    def generate_insights(self, df: pd.DataFrame, question: str, analysis_type: dict) -> str:
        """Genera insights intelligenti sui dati"""
        if df is None or df.empty:
            return "Nessun dato disponibile per l'analisi."
        
        # Prepara i dati per l'AI
        if len(df) > 10:
            sample_data = df.head(10).to_string(index=False)
            data_summary = f"Mostrando prime 10 righe di {len(df)} totali:\n{sample_data}"
        else:
            data_summary = df.to_string(index=False)
        
        # Determina il tipo di insight richiesto
        if analysis_type['is_forecasting']:
            insight_type = "Fornisci raccomandazioni strategiche e previsioni basate sui dati"
        else:
            insight_type = "Fornisci insights analitici e spiegazioni dei pattern nei dati"
        
        prompt = f"""
Domanda utente: "{question}"

Dati ottenuti:
{data_summary}

{insight_type}. Rispondi in modo:
- Chiaro e actionable
- Con numeri specifici dai dati
- Se è una previsione, spiega il ragionamento
- Se è un'analisi, evidenzia i pattern principali
- Massimo 3-4 punti chiave

Rispondi in italiano:
"""

        response = self.oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analista di business che fornisce insights chiari e actionable"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def run_forecast(self, df: pd.DataFrame, question: str) -> None:
        """Esegue forecasting avanzato quando richiesto"""
        if df is None or df.empty or len(df.columns) < 2:
            return
            
        try:
            # Trova colonne date e numeriche
            date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time'])]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not date_cols or not numeric_cols:
                return
                
            date_col = date_cols[0]
            metric_col = numeric_cols[0]
            
            # Prepara dati per Prophet
            df_clean = df[[date_col, metric_col]].dropna()
            df_clean[date_col] = pd.to_datetime(df_clean[date_col])
            df_clean = df_clean.sort_values(date_col)
            
            if len(df_clean) < 10:
                st.warning("⚠️ Troppo pochi dati per un forecast accurato")
                return
            
            # Crea DataFrame per Prophet
            prophet_df = pd.DataFrame({
                'ds': df_clean[date_col],
                'y': df_clean[metric_col]
            })
            
            # Addestra modello
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=len(prophet_df) > 365
            )
            model.fit(prophet_df)
            
            # Determina periodo di forecast
            if any(word in question.lower() for word in ['mese', 'month']):
                periods = 30
                period_name = "prossimo mese"
            elif any(word in question.lower() for word in ['anno', 'year', '2025']):
                periods = 365
                period_name = "prossimo anno"
            else:
                periods = 60
                period_name = "prossimi 2 mesi"
            
            # Genera forecast
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Visualizza risultati
            st.subheader("📈 Previsione")
            
            # Grafico forecast
            fig = go.Figure()
            
            # Dati storici
            fig.add_trace(go.Scatter(
                x=prophet_df['ds'], y=prophet_df['y'],
                mode='lines+markers', name='Dati Storici',
                line=dict(color='blue')
            ))
            
            # Previsioni
            future_only = forecast.tail(periods)
            fig.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat'],
                mode='lines', name='Previsione',
                line=dict(color='red', dash='dash')
            ))
            
            # Intervallo di confidenza
            fig.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat_upper'],
                fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat_lower'],
                fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                name='Intervallo di confidenza', fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig.update_layout(
                title=f"Previsione {metric_col} per {period_name}",
                xaxis_title="Data",
                yaxis_title=metric_col
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiche previsione
            total_forecast = future_only['yhat'].sum()
            st.metric("Totale Previsto", f"{total_forecast:,.0f}")
            
        except Exception as e:
            st.error(f"Errore nel forecasting: {e}")

    def process_question(self, question: str):
        """Processa una domanda dall'inizio alla fine"""
        
        # 1. Analizza la domanda
        analysis_type = self.analyze_question(question)
        
        # 2. Genera SQL
        with st.spinner("🔍 Generando query SQL..."):
            sql = self.generate_sql(question, analysis_type)
        
        # 3. Mostra SQL generato
        with st.expander("🔧 SQL Generato", expanded=False):
            st.code(sql, language="sql")
        
        # 4. Esegui query
        with st.spinner("⚡ Eseguendo query..."):
            df = self.execute_sql(sql)
        
        if df is None or df.empty:
            st.error("❌ Nessun risultato trovato")
            return
        
        # 5. Mostra risultati
        st.subheader("📊 Risultati")
        st.dataframe(df)
        
        # 6. Crea visualizzazioni
        self.create_visualization(df, question)
        
        # 7. Esegui forecast se necessario
        if analysis_type['is_forecasting']:
            self.run_forecast(df, question)
        
        # 8. Genera insights
        with st.spinner("🧠 Generando insights..."):
            insights = self.generate_insights(df, question, analysis_type)
        
        st.subheader("💡 Insights")
        st.markdown(insights)


def main():
    st.title("🤖 Smart Analytics Assistant")
    st.markdown("*Fai qualsiasi domanda sui tuoi dati - risponderò con analisi e previsioni intelligenti*")
    
    # Esempi di domande
    with st.expander("💭 Esempi di domande che puoi fare"):
        st.markdown("""
        **Analisi Descrittive:**
        - Quanti ordini ho fatto questo mese?
        - Chi è il cliente che spende di più?
        - Qual è la città con più ordini?
        - Come va il canale online vs offline?
        
        **Analisi Temporali:**
        - Mostrami il trend degli ordini negli ultimi 6 mesi
        - Qual è la stagionalità delle vendite?
        - Come cresce il revenue nel tempo?
        
        **Previsioni e Strategia:**
        - In quale città dovrei investire e perché?
        - Quanti ordini farò il prossimo mese?
        - Quale canale ha più potenziale?
        - Dammi una strategia per aumentare le vendite
        """)
    
    # Input principale
    question = st.text_input(
        "💬 Fai la tua domanda:",
        placeholder="es. Quanti ordini ho fatto questo mese?"
    )
    
    if st.button("🚀 Analizza", type="primary"):
        if question:
            st.session_state.query_history.append(question)
            
            engine = SmartAnalyticsEngine()
            engine.process_question(question)
        else:
            st.warning("⚠️ Inserisci una domanda!")
    
    # Sidebar con cronologia e esempi
    with st.sidebar:
        st.header("📝 Cronologia")
        if st.session_state.query_history:
            for i, q in enumerate(reversed(st.session_state.query_history[-5:])):
                if st.button(f"🔄 {q[:40]}...", key=f"history_{i}"):
                    engine = SmartAnalyticsEngine()
                    engine.process_question(q)
        else:
            st.write("*Nessuna query ancora*")
        
        st.divider()
        
        # Esempi di domande analitiche
        st.header("📊 Domande Analitiche")
        st.write("*Clicca per usare come esempio*")
        
        analytical_questions = [
            "Quanti ordini ho fatto questo mese?",
            "Chi è il cliente che spende di più?",
            "Qual è la città con più ordini?",
            "Quale canale performa meglio?",
            "Mostrami il revenue per paese",
            "Qual è l'ordine medio per città?",
            "Chi sono i top 10 clienti per spesa?",
            "Quanti ordini per canale questo anno?",
            "Qual è la distribuzione geografica delle vendite?",
            "Mostrami le statistiche di questo trimestre"
        ]
        
        for q in analytical_questions:
            if st.button(q, key=f"analytical_{q[:20]}", use_container_width=True):
                engine = SmartAnalyticsEngine()
                engine.process_question(q)
        
        st.divider()
        
        # Esempi di domande predittive
        st.header("🔮 Domande Predittive")
        st.write("*Strategie e previsioni*")
        
        forecasting_questions = [
            "In quale città dovrei investire e perché?",
            "Quanti ordini farò il prossimo mese?",
            "Quale canale ha più potenziale futuro?",
            "Dammi previsioni di revenue per dicembre",
            "Dove dovrei aprire il prossimo negozio?",
            "Quale strategia per aumentare le vendite?",
            "Prevedi il trend dei prossimi 3 mesi",
            "Su quale paese dovrei concentrarmi?",
            "Raccomandazioni per il Q4 2025",
            "Analisi predittiva per fine anno"
        ]
        
        for q in forecasting_questions:
            if st.button(q, key=f"forecasting_{q[:20]}", use_container_width=True):
                engine = SmartAnalyticsEngine()
                engine.process_question(q)

if __name__ == "__main__":
    main()