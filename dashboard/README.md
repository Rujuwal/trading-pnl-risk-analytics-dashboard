# Python Dashboard

Run the interactive dashboard from the project root:

```powershell
python -m streamlit run dashboard/app.py
```

The dashboard reads generated analytics exports from `outputs/dashboard`. If those files do not exist, it creates the sample database and exports automatically.

Dashboard tabs:

- Overview
- Strategy Performance
- Current Risk
- Trade Quality

