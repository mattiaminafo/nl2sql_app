import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from openai import OpenAI
import re
import logging
from prophet import Prophet    # Assicurati di avere prophet in requirements.txt

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
        sys = (
            "You are an analytics assistant. "
            "Classify the user's request into one of: top_n, distribution, correlation, forecast, raw, summary."
        )
        resp = self.oa.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":f"Request: \"{nl}\""}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip().lower()

    def generate_sql(self, nl: str) -> str:
        prompt = (
            f"You are a BigQuery SQL expert. Given the request:\n\"{nl}\"\n"
            f"Generate only the SQL (no explanation) on table `{self.dataset}`."
        )
        resp = self.oa.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"SQL expert"},
                {"role":"user","content":prompt}
            ],
            temperature=0.1
        )
        sql = resp.choices[0].message.content
        return re.sub(r'```.*?\n|```', '', sql).strip()

    def run(self, user_query: str):
        analysis = self.classify_request(user_query)
        sql = self.generate_sql(user_query)

        # Show SQL
        st.markdown("### Generated SQL Query")
        st.code(sql, language="sql")

        # Execute
        df = self.bq.query(sql).result().to_dataframe()
        if df.empty:
            st.warning("Nessun risultato.")
            return

        # Dispatch based on classification
        if analysis == "top_n":
            c1, c2 = df.columns[:2]
            lines = [f"Top results by {c2}:"]
            for i, row in df.iterrows():
                lines.append(f"{i+1}. {row[c1]} → {row[c2]:,.0f}")
            st.markdown("\n".join(lines))

        elif analysis == "distribution":
            col = df.columns[0]
            st.bar_chart(df[col].value_counts().head(10))

        elif analysis == "correlation":
            st.dataframe(df.corr())

        elif analysis == "forecast":
            df_ts = pd.DataFrame({"ds": pd.to_datetime(df.iloc[:,0]), "y": df.iloc[:,1]})
            m = Prophet()
            m.fit(df_ts)
            future = m.make_future_dataframe(periods=30)
            fc = m.predict(future)
            st.line_chart(fc.set_index("ds")[["yhat","yhat_lower","yhat_upper"]])

        elif analysis == "raw":
            st.dataframe(df)

        else:  # summary
            st.table(df.head(5))
            st.markdown(f"Found **{len(df)}** rows. Showing top 5.")

def main():
    st.title("🔍 Natural Language Analytics Interface")
    qu = st.text_input("Ask anything (analytics/statistics/forecast)...")
    if st.button("Run"):
        st.session_state.query_history.append(qu)
        with st.spinner("Thinking…"):
            engine = NL2AnalyticsEngine()
            engine.run(qu)

    if st.session_state.query_history:
        st.sidebar.header("Recent Queries")
        for q in st.session_state.query_history[-5:]:
            st.sidebar.write(q)

if __name__ == "__main__":
    main()
