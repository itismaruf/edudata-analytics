create extension if not exists pgcrypto;

create table if not exists projects (
    id uuid primary key default gen_random_uuid(),
    user_id text null,
    name text not null,
    description text null,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create table if not exists datasets (
    id uuid primary key default gen_random_uuid(),
    project_id uuid references projects(id) on delete cascade,
    original_filename text not null,
    file_path text not null,
    rows_count integer,
    columns_count integer,
    created_at timestamptz default now()
);

create table if not exists educational_schemas (
    id uuid primary key default gen_random_uuid(),
    dataset_id uuid references datasets(id) on delete cascade unique,
    student_col text null,
    student_name_col text null,
    group_col text null,
    course_col text null,
    score_col text null,
    attendance_col text null,
    final_score_col text null,
    status_col text null,
    schema_json jsonb null,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create table if not exists analysis_results (
    id uuid primary key default gen_random_uuid(),
    dataset_id uuid references datasets(id) on delete cascade,
    result_type text not null,
    result_json jsonb not null,
    created_at timestamptz default now()
);

create table if not exists reports (
    id uuid primary key default gen_random_uuid(),
    dataset_id uuid references datasets(id) on delete cascade,
    report_title text not null,
    report_markdown text null,
    report_path text null,
    created_at timestamptz default now()
);

create index if not exists idx_projects_user_id on projects(user_id);
create index if not exists idx_datasets_project_id on datasets(project_id);
create index if not exists idx_analysis_results_dataset_id on analysis_results(dataset_id);
create index if not exists idx_analysis_results_result_type on analysis_results(result_type);
create index if not exists idx_reports_dataset_id on reports(dataset_id);

create or replace function set_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists trg_projects_updated_at on projects;
create trigger trg_projects_updated_at
before update on projects
for each row execute function set_updated_at();

drop trigger if exists trg_educational_schemas_updated_at on educational_schemas;
create trigger trg_educational_schemas_updated_at
before update on educational_schemas
for each row execute function set_updated_at();

-- Create Storage bucket manually in Supabase dashboard:
-- Bucket name: datasets
