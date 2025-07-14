Natural Language to SQL Query Interface
Un'applicazione Streamlit che permette di interrogare un dataset BigQuery utilizzando domande in linguaggio naturale (inglese). L'applicazione converte automaticamente le domande in query SQL e restituisce risposte comprensibili.

🚀 Funzionalità
Interfaccia intuitiva: Fai domande in inglese naturale
Conversione automatica: Trasforma domande in query SQL usando OpenAI
Connessione BigQuery: Esegue query direttamente sul dataset
Risposte naturali: Converte i risultati in risposte comprensibili
Cronologia: Mantiene traccia delle domande recenti
📋 Prerequisiti
Account Google Cloud con BigQuery abilitato
Service Account con permessi di lettura su BigQuery
OpenAI API Key
Account Streamlit Cloud (per il deployment)
🛠️ Setup Locale
Clona il repository:
bash
git clone <your-repo-url>
cd <your-repo-name>
Installa le dipendenze:
bash
pip install -r requirements.txt
Configura i secrets:
Crea la cartella .streamlit/ nella root del progetto
Copia secrets.toml template in .streamlit/secrets.toml
Compila con le tue credenziali (vedi sezione Configurazione)
Avvia l'applicazione:
bash
streamlit run app.py
🔧 Configurazione
1. Service Account BigQuery
Vai su Google Cloud Console
Naviga a IAM & Admin > Service Accounts
Trova il service account: bq-access-app@planar-flux-465609-e1.iam.gserviceaccount.com
Genera una nuova chiave JSON
Copia il contenuto del file JSON nel file secrets.toml
2. OpenAI API Key
La tua chiave è già configurata nel template. Se devi cambiarla:

Vai su OpenAI Platform
Genera una nuova API key
Sostituiscila nel file secrets.toml
🌐 Deployment su Streamlit Cloud
Opzione 1: Deployment Automatico
Push del codice:
bash
git add .
git commit -m "Initial commit"
git push origin main
Configura Streamlit Cloud:
Vai su share.streamlit.io
Connetti il tuo repository GitHub
Seleziona branch main e file app.py
Configura i secrets:
Nella dashboard di Streamlit Cloud, vai su "Secrets"
Copia il contenuto del tuo secrets.toml compilato
Salva le configurazioni
Opzione 2: Deployment Manuale
Crea l'app su Streamlit:
Vai su share.streamlit.io
Clicca "New app"
Seleziona il repository GitHub
Scegli app.py come main file
Configura secrets:
Vai nelle impostazioni dell'app
Sezione "Secrets"
Incolla il contenuto del tuo secrets.toml
📊 Dataset Schema
Il dataset planar-flux-465609-e1.locatify_data.brand_orders contiene:

order_id: ID univoco ordine
channel: Canale di vendita
order_date: Data ordine
city: Città del cliente
country_code: Codice paese
total_eur: Totale in euro
customer_id: ID cliente
E altri campi...
💡 Esempi di Domande
Which city has the most orders?
Which channel does the most orders in July?
What is the average order value?
How many orders were placed last month?
Which country has the highest total revenue?
Show me the top 5 cities by order count
What is the total revenue for Italy?
🔍 Come Funziona
Input: L'utente inserisce una domanda in inglese
Conversione: OpenAI GPT-3.5 converte la domanda in SQL
Esecuzione: La query viene eseguita su BigQuery
Formatting: I risultati vengono convertiti in linguaggio naturale
Output: L'utente riceve una risposta comprensibile
🚨 Troubleshooting
Errori comuni:
"BigQuery credentials not found":
Verifica che il file secrets.toml sia nella cartella .streamlit/
Controlla che le credenziali del service account siano corrette
"OpenAI API key not found":
Verifica che la chiave OpenAI sia presente in secrets.toml
Controlla che la chiave sia valida e abbia crediti
"Query execution failed":
Controlla che il service account abbia i permessi di lettura su BigQuery
Verifica che il dataset sia accessibile
Debug:
Per vedere i log dettagliati, esegui:

bash
streamlit run app.py --logger.level=debug
🔐 Sicurezza
Mai committare file secrets.toml nel repository
Usa sempre Streamlit secrets per le credenziali in produzione
Monitora l'uso delle API OpenAI per controllare i costi
📝 Limitazioni
Supporta solo domande in inglese
Limitato a 100 risultati per query
Dipende dalla qualità del modello OpenAI per la conversione NL2SQL
Richiede connessione internet per funzionare
🤝 Contribuire
Fork del repository
Crea un branch per la tua feature
Fai commit delle modifiche
Pusha il branch
Crea una Pull Request
📞 Supporto
Per problemi o domande:

Apri un issue nel repository
Controlla la documentazione di Streamlit
Verifica la documentazione di BigQuery
Creato con ❤️ usando Streamlit, OpenAI e BigQuery

