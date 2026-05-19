# Supabase Setup

This app can run without Supabase. Supabase is only used as a persistence layer for projects, dataset files, metadata, educational schemas, analysis results, and reports.

## 1. Create a Supabase Project

Create a new project at https://supabase.com and open the project dashboard.

## 2. Create Database Tables

Open the Supabase SQL editor and run the contents of:

```text
supabase_schema.sql
```

## 3. Create Storage Bucket

Create a Storage bucket manually:

```text
Bucket name: datasets
```

For local testing, configure bucket policies according to your security model. If you use an anon key, make sure uploads/downloads are allowed by your Storage policies. For server-side trusted deployments, use a service role key only in secure server secrets.

## 4. Add Streamlit Secrets

Add these values to `.streamlit/secrets.toml` or provide them as environment variables:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-anon-or-service-key"
```

Use the project base URL for `SUPABASE_URL`. If you copied a REST endpoint ending with `/rest/v1/`, remove that suffix.

## 5. Install Dependency

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 6. Run App

```bash
streamlit run Home.py
```

Then open the `Projects` page to create a project, save uploaded datasets, and restore saved datasets later.
