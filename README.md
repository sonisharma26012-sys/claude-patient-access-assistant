# Claude-Powered Patient Access & No-Show Prevention Assistant

## Why this project fits Claude Corps
This project shows how Claude can help a healthcare or public-interest organization turn operational data into useful decisions. The app analyzes real appointment records, identifies no-show risk patterns, and generates stakeholder-ready summaries for non-technical clinic or public-health teams.

## Dataset used
**Dataset:** Medical Appointment No-Shows Dataset / KaggleV2-May-2016.csv  
**Rows used:** 110,526 valid appointment records after removing one invalid age record  
**Columns:** PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age, Neighbourhood, Scholarship, Hypertension, Diabetes, Alcoholism, Handicap, SMS_received, No-show

## Main metrics produced
- Total appointments
- Unique patients
- Overall no-show rate
- Average and median wait time
- No-show rate by age group
- No-show rate by wait-time group
- SMS reminder analysis
- Chronic condition analysis: diabetes and hypertension
- High-risk neighborhood ranking
- High-risk patient access segments
- Monthly no-show trend

## How Claude is used
The Streamlit app has a **Generate Executive Summary** button. If you add an Anthropic API key, Claude creates a clinic-leader summary using the calculated metrics. If you do not add a key, the app still opens and uses a local fallback summary so you can demo it on your laptop.

## How to run on Windows laptop
1. Install Python from python.org.
2. Download and unzip this project folder.
3. Double-click `run_app_windows.bat`.
4. Your browser should open the Streamlit app.

If double-click does not work:
```bash
cd patient_access_project/app
pip install -r requirements.txt
streamlit run app.py
```

## How to run on Mac/Linux
```bash
cd patient_access_project/app
pip3 install -r requirements.txt
streamlit run app.py
```

## Optional: add Claude API key
Create a file named `.env` inside the `app` folder or set this environment variable:
```bash
ANTHROPIC_API_KEY=your_key_here
```
The app works even without this.

## Project structure
```text
patient_access_project/
  app/
    app.py
    requirements.txt
    .env.example
  data/
    KaggleV2-May-2016.csv
    appointments_cleaned.csv
  outputs/
    project_metrics.csv
    project_metrics.json
    age_group_no_show.csv
    wait_group_no_show.csv
    top_neighborhood_risk.csv
    high_risk_segments.csv
    monthly_trends.csv
  notebooks/
    Patient_Access_Claude_Project_Colab.ipynb
    Patient_Access_Claude_Project_Colab.py
  docs/
    resume_bullets.txt
    claude_corps_project_story.txt
    step_by_step_how_created.txt
```

## Demo talking points
1. I used a real healthcare appointment dataset with 110K+ records.
2. I cleaned dates, corrected columns, engineered wait-time and no-show features.
3. I calculated operational metrics such as no-show rate, wait-time risk, SMS reminder patterns, and high-risk segments.
4. I built a Streamlit app for non-technical users.
5. I integrated Claude-style executive reporting so clinic leaders can act on the metrics without writing SQL.
