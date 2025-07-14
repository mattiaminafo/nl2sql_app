import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import BadRequest
from openai import OpenAI
import re
import logging
from prophet import Prophet  # assicurati di aggiungere "prophet>=1.0" a requirements.txt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="NL to SQL Analytics", page_icon="🔍", layout="wide")

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

class NL2AnalyticsEngine:
    def __init__(self):
        self.dataset = "planar-flux-465609-e1.locatify_data.brand_orders"
        self.bq = None
        self.oa = None
        self._setup_clients()

    def _setup_clients(self):
        project = st.secrets["project_id"]
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        self.bq = bigquery.Client(project=project, credentials=creds)
        self.oa = OpenAI(api_key=st.secrets["openai_api_key"])

    def classify_request(self, nl: str) -> str:
        system_prompt = (
            "You are an analytics assistant. "
            "Classify the user's request into exactly one of: "
            "top_n, distribution, correlation, forecast, raw, summary."
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
            f"Generate only the SQL (no explanations) on table `{self.dataset}`."
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

    def run(self, user_query: str):
        analysis = self.classify_request(user_query)
        sql = self.generate_sql(user_query)

        # Display generated SQL
        st.markdown("### Generated SQL Query")
        st.code(sql, language="sql")

        df = None
        try:
            job = self.bq.query(sql)
            df = job.result().to_dataframe()
        except BadRequest as e:
            msg = e.message or str(e)
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
                    st.warning(f"Colonna `{bad_col}` non trovata; sostituisco con `{corrected}` e ritento.")
                    sql_fixed = re.sub(rf"\b{bad_col}\b", corrected, sql, flags=re.IGNORECASE)
                    st.markdown("### SQL Corretto")
                    st.code(sql_fixed, language="sql")
                    try:
                        job = self.bq.query(sql_fixed)
                        df = job.result().to_dataframe()
                    except Exception as e2:
                        st.error(f"Errore con SQL corretto: {e2}")
                        return
                else:
                    st.error(f"Errore SQL: {msg}")
                    return
            else:
                st.error(f"Errore SQL: {msg}")
                return
        except Exception as e:
            st.error(f"Errore esecuzione query: {e}")
            return

        if df is None or df.empty:
            st.warning("Nessun risultato.")
            return

        # Dispatch
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
            st.line_chart(forecast.set_index("ds")["yhat", "yhat_lower", "yhat_upper"])
            st.markdown("Forecast for next 30 periods")

        elif analysis == "raw":
            st.markdown("### Raw Data")
            st.dataframe(df)

        else:
            st.markdown("### Summary (top 5 rows)")
            st.table(df.head(5))
            st.markdown(f"Found **{len(df)}** rows total.")


def main():
    st.title("🔍 Natural Language Analytics Interface")
    user_query = st.text_input("Ask anything (analytics/statistics/forecast)...")
    if st.button("Run"):
        st.session_state.query_history.append(user_query)
        with st.spinner("Processing your request…"):
            engine = NL2AnalyticsEngine()
            engine.run(user_query)

    if st.session_state.query_history:
        st.sidebar.header("Recent Queries")
        for q in st.session_state.query_history[-5:]:
            st.sidebar.write(f"- {q}")

if __name__ == "__main__":
    main()
