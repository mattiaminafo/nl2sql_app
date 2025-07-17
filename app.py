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
        self.project_id = "planar-flux-465609-e1"
        self.dataset_id = "locatify_dataset_1"
        self.table_id = "brand_orders"
        self.full_table_name = f"`{self.project_id}.{self.dataset_id}.{self.table_id}`"
        
        # Database schema
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
        """Analyze the question and determine the type of response needed"""
        
        # Keywords to identify analysis type
        forecasting_keywords = [
            'forecast', 'predict', 'future', 'next', 'should invest', 'invest',
            'recommend', 'recommendation', 'strategy', 'best', 'which city',
            'where', 'why', 'motivation', 'prevision', 'projection', 'trend',
            'growth', 'potential', 'optimize', 'improve', 'suggestion'
        ]
        
        time_keywords = [
            'trend', 'over time', 'growth', 'seasonal', 'monthly', 'daily',
            'weekly', 'yearly', 'evolution', 'change', 'pattern', 'timeline'
        ]
        
        q_lower = question.lower()
        
        # Determine analysis type
        is_forecasting = any(keyword in q_lower for keyword in forecasting_keywords)
        is_time_series = any(keyword in q_lower for keyword in time_keywords) or is_forecasting
        
        return {
            'is_forecasting': is_forecasting,
            'is_time_series': is_time_series,
            'original_question': question
        }

    def generate_sql(self, question: str, analysis_type: dict) -> str:
        """Generate intelligent SQL based on the question"""
        
        # Create schema description
        schema_desc = "\n".join([f"- {col}: {dtype}" for col, dtype in self.schema.items()])
        
        # Specific context for analysis type
        if analysis_type['is_forecasting']:
            context = (
                "The user wants forecasts or strategic recommendations. "
                "Generate SQL to get historical data that allows predictive analysis. "
                "Always include a time dimension (DATE(order_date_timestamp)) when possible."
            )
        elif analysis_type['is_time_series']:
            context = (
                "The user wants temporal analysis or trends. "
                "Generate SQL that includes DATE(order_date_timestamp) for time-based analysis."
            )
        else:
            context = (
                "The user wants descriptive or statistical analysis. "
                "Generate SQL to directly answer the question."
            )

        prompt = f"""
You are a BigQuery SQL expert. The user asked: "{question}"

CONTEXT: {context}

TABLE SCHEMA {self.full_table_name}:
{schema_desc}

IMPORTANT RULES:
- Use EXACTLY the column names from the schema
- For dates use: DATE(order_date_timestamp)
- For orders use: COUNT(*) or COUNT(order_id) 
- For revenue use: SUM(total_eur)
- For customers use: customer_id
- Always use appropriate ORDER BY
- If it's temporal analysis, always include the date

EXAMPLES:
- "orders this month" → WHERE DATE(order_date_timestamp) >= DATE_TRUNC(CURRENT_DATE(), MONTH)
- "top spending customer" → GROUP BY customer_id ORDER BY SUM(total_eur) DESC
- "monthly order trend" → GROUP BY DATE_TRUNC(DATE(order_date_timestamp), MONTH)

Generate ONLY the SQL, no explanations:
"""

        response = self.oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "SQL expert who generates precise and optimized queries"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        sql = response.choices[0].message.content
        return re.sub(r'```.*?\n|```', '', sql).strip()

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL with intelligent error handling"""
        try:
            job = self.bq.query(sql)
            return job.result().to_dataframe()
        except BadRequest as e:
            msg = str(e)
            
            # Auto-fix common errors
            if "Unrecognized name" in msg:
                # Extract wrong column name
                match = re.search(r"Unrecognized name: (\w+)", msg)
                if match:
                    wrong_col = match.group(1)
                    
                    # Common mappings
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
                        st.warning(f"🔧 Fixed: {wrong_col} → {col_fixes[wrong_col]}")
                        st.code(fixed_sql, language="sql")
                        return self.execute_sql(fixed_sql)
            
            st.error(f"❌ SQL Error: {msg}")
            return None
        except Exception as e:
            st.error(f"❌ Execution Error: {e}")
            return None

    def create_visualization(self, df: pd.DataFrame, question: str) -> None:
        """Create intelligent visualizations based on data"""
        if df is None or df.empty:
            return
            
        # Identify temporal columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # If there are dates, create temporal charts
        if date_cols and numeric_cols:
            date_col = date_cols[0]
            if len(numeric_cols) > 0:
                df[date_col] = pd.to_datetime(df[date_col])
                df_sorted = df.sort_values(date_col)
                
                for num_col in numeric_cols[:2]:  # Max 2 metrics
                    fig = px.line(df_sorted, x=date_col, y=num_col, 
                                title=f"📈 {num_col.title()} Over Time")
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title=num_col.title(),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Charts for top rankings
        elif len(df.columns) >= 2 and len(df) <= 20:
            if df.dtypes.iloc[1] in ['int64', 'float64']:
                fig = px.bar(df.head(10), x=df.columns[0], y=df.columns[1],
                           title=f"📊 Top {df.columns[0]} by {df.columns[1]}")
                fig.update_layout(
                    xaxis_title=df.columns[0].title(),
                    yaxis_title=df.columns[1].title(),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

    def generate_insights(self, df: pd.DataFrame, question: str, analysis_type: dict) -> str:
        """Generate intelligent insights about the data"""
        if df is None or df.empty:
            return "No data available for analysis."
        
        # Prepare data for AI
        if len(df) > 10:
            sample_data = df.head(10).to_string(index=False)
            data_summary = f"Showing first 10 rows of {len(df)} total:\n{sample_data}"
        else:
            data_summary = df.to_string(index=False)
        
        # Determine insight type required
        if analysis_type['is_forecasting']:
            insight_type = "Provide strategic recommendations and forecasts based on the data"
        else:
            insight_type = "Provide analytical insights and explanations of patterns in the data"
        
        prompt = f"""
User question: "{question}"

Data obtained:
{data_summary}

{insight_type}. Respond in a way that is:
- Clear and actionable
- With specific numbers from the data
- If it's a forecast, explain the reasoning
- If it's an analysis, highlight main patterns
- Maximum 3-4 key points

Respond in English:
"""

        response = self.oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Business analyst who provides clear and actionable insights"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def run_forecast(self, df: pd.DataFrame, question: str) -> None:
        """Run advanced forecasting when requested"""
        if df is None or df.empty or len(df.columns) < 2:
            return
            
        try:
            # Find date and numeric columns
            date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time'])]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not date_cols or not numeric_cols:
                return
                
            date_col = date_cols[0]
            metric_col = numeric_cols[0]
            
            # Prepare data for Prophet
            df_clean = df[[date_col, metric_col]].dropna()
            df_clean[date_col] = pd.to_datetime(df_clean[date_col])
            df_clean = df_clean.sort_values(date_col)
            
            if len(df_clean) < 10:
                st.warning("⚠️ Too few data points for accurate forecasting")
                return
            
            # Create DataFrame for Prophet
            prophet_df = pd.DataFrame({
                'ds': df_clean[date_col],
                'y': df_clean[metric_col]
            })
            
            # Train model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=len(prophet_df) > 365
            )
            model.fit(prophet_df)
            
            # Determine forecast period
            if any(word in question.lower() for word in ['month', 'monthly']):
                periods = 30
                period_name = "next month"
            elif any(word in question.lower() for word in ['year', 'yearly', '2025']):
                periods = 365
                period_name = "next year"
            else:
                periods = 60
                period_name = "next 2 months"
            
            # Generate forecast
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Display results
            st.subheader("📈 Forecast")
            
            # Forecast chart with Plotly
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=prophet_df['ds'], y=prophet_df['y'],
                mode='lines+markers', name='Historical Data',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            
            # Predictions
            future_only = forecast.tail(periods)
            fig.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat'],
                mode='lines', name='Forecast',
                line=dict(color='#ff7f0e', dash='dash', width=3)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat_upper'],
                fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=future_only['ds'], y=future_only['yhat_lower'],
                fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                name='Confidence Interval', 
                fillcolor='rgba(255,127,14,0.2)',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title=f"🔮 {metric_col} Forecast for {period_name}",
                xaxis_title="Date",
                yaxis_title=metric_col.title(),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast statistics
            total_forecast = future_only['yhat'].sum()
            avg_forecast = future_only['yhat'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Predicted", f"{total_forecast:,.0f}")
            with col2:
                st.metric("Daily Average", f"{avg_forecast:,.1f}")
            
            # Show forecast details
            with st.expander("📋 Forecast Details"):
                future_display = future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                future_display.columns = ['Date', 'Forecast', 'Min', 'Max']
                future_display['Date'] = future_display['Date'].dt.strftime('%Y-%m-%d')
                future_display['Forecast'] = future_display['Forecast'].round(1)
                future_display['Min'] = future_display['Min'].round(1)
                future_display['Max'] = future_display['Max'].round(1)
                st.dataframe(future_display, use_container_width=True)
                
        except Exception as e:
            st.error(f"Forecasting error: {e}")

    def process_question(self, question: str):
        """Process a question from start to finish"""
        
        # 1. Analyze the question
        analysis_type = self.analyze_question(question)
        
        # 2. Generate SQL
        with st.spinner("🔍 Generating SQL query..."):
            sql = self.generate_sql(question, analysis_type)
        
        # 3. Show generated SQL
        with st.expander("🔧 Generated SQL", expanded=False):
            st.code(sql, language="sql")
        
        # 4. Execute query
        with st.spinner("⚡ Executing query..."):
            df = self.execute_sql(sql)
        
        if df is None or df.empty:
            st.error("❌ No results found")
            return
        
        # 5. Show results
        st.subheader("📊 Results")
        st.dataframe(df)
        
        # 6. Create visualizations
        self.create_visualization(df, question)
        
        # 7. Run forecast if necessary
        if analysis_type['is_forecasting']:
            self.run_forecast(df, question)
        
        # 8. Generate insights
        with st.spinner("🧠 Generating insights..."):
            insights = self.generate_insights(df, question, analysis_type)
        
        st.subheader("💡 Insights")
        st.markdown(insights)


def main():
    st.title("🤖 Smart Analytics Assistant")
    st.markdown("*Ask any question about your data - I'll respond with intelligent analysis and predictions*")
    
    # Question examples
    with st.expander("💭 Example questions you can ask"):
        st.markdown("""
        **Descriptive Analytics:**
        - How many orders did I make this month?
        - Who is my top spending customer?
        - Which city has the most orders?
        - How is online vs offline channel performing?
        
        **Temporal Analysis:**
        - Show me the order trend over the last 6 months
        - What's the seasonality of my sales?
        - How does revenue grow over time?
        
        **Forecasting & Strategy:**
        - Which city should I invest in and why?
        - How many orders will I make next month?
        - Which channel has the most potential?
        - Give me revenue predictions for December
        - Where should I open my next store?
        - What strategy to increase sales?
        """)
    
    # Main input
    question = st.text_input(
        "💬 Ask your question:",
        placeholder="e.g., How many orders did I make this month?"
    )
    
    if st.button("🚀 Analyze", type="primary"):
        if question:
            st.session_state.query_history.append(question)
            
            engine = SmartAnalyticsEngine()
            engine.process_question(question)
        else:
            st.warning("⚠️ Please enter a question!")
    
    # Sidebar with history and examples
    with st.sidebar:
        st.header("📝 History")
        if st.session_state.query_history:
            for i, q in enumerate(reversed(st.session_state.query_history[-5:])):
                if st.button(f"🔄 {q[:40]}...", key=f"history_{i}"):
                    engine = SmartAnalyticsEngine()
                    engine.process_question(q)
        else:
            st.write("*No queries yet*")
        
        st.divider()
        
        # Analytical question examples
        st.header("📊 Analytical Questions")
        st.write("*Click to use as example*")
        
        analytical_questions = [
            "How many orders did I make this month?",
            "Who is my top spending customer?",
            "Which city has the most orders?",
            "Which channel performs better?",
            "Show me revenue by country",
            "What's the average order value by city?",
            "Who are my top 10 customers by spend?",
            "How many orders per channel this year?",
            "What's the geographic distribution of sales?",
            "Show me this quarter's statistics"
        ]
        
        for q in analytical_questions:
            if st.button(q, key=f"analytical_{q[:20]}", use_container_width=True):
                engine = SmartAnalyticsEngine()
                engine.process_question(q)
        
        st.divider()
        
        # Predictive question examples
        st.header("🔮 Predictive Questions")
        st.write("*Strategy and forecasting*")
        
        forecasting_questions = [
            "Which city should I invest in and why?",
            "How many orders will I make next month?",
            "Which channel has the most potential?",
            "Give me revenue predictions for December",
            "Where should I open my next store?",
            "What strategy to increase sales?",
            "Predict the trend for next 3 months",
            "Which country should I focus on?",
            "Q4 2025 recommendations",
            "End of year predictive analysis"
        ]
        
        for q in forecasting_questions:
            if st.button(q, key=f"forecasting_{q[:20]}", use_container_width=True):
                engine = SmartAnalyticsEngine()
                engine.process_question(q)

if __name__ == "__main__":
    main()