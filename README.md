# Store-Locked Metrics Dashboard (Streamlit + Supabase)

One codebase → many store deployments.  
Each deployment points to the same repo/branch but sets a different `STORE_PC` env var (e.g., `301290`).  
The app auto-selects the **latest date** for that store and renders:

- 💼 Labor % to Sales — Weekly / MTD / QTD / YTD  
- 💵 Sales % change — Weekly / MTD / QTD / YTD (vs previous period)  
- 👥 Guest count % change — Weekly / MTD / QTD / YTD  
- 🧾 Void counts — Weekly / MTD / QTD / YTD  
- 🚗 HME metrics (weighted by cars) — period deltas + daypart breakdown (last 7 days)

## Repo layout

