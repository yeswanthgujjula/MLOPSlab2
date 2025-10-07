from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta
import sys, os, pickle

# ✅ Enable pickled Python objects in XComs
conf.set('core', 'enable_xcom_pickling', 'True')

# ✅ Add the src directory so Airflow can find lab2.py
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.lab2 import load_data, data_preprocessing, build_save_model, load_model_elbow

default_args = {
    'owner': 'yashwanth',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='iris_clustering_dag',
    default_args=default_args,
    description='Iris dataset KMeans clustering pipeline (MLOps-ready)',
    schedule_interval=None,
    start_date=datetime(2025, 10, 5),
    catchup=False,
    tags=['iris', 'mlops', 'clustering'],
) as dag:

    # 1️⃣ Load data
    t1 = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    # 2️⃣ Preprocess data (direct pull from XCom)
    t2 = PythonOperator(
        task_id='preprocess_data',
        python_callable=lambda ti: data_preprocessing(
            ti.xcom_pull(task_ids='load_data')
        ),
    )

    # 3️⃣ Train & save model
    t3 = PythonOperator(
        task_id='build_model',
        python_callable=lambda ti: build_save_model(
            ti.xcom_pull(task_ids='preprocess_data')
        ),
    )

    # 4️⃣ Generate elbow plot
    t4 = PythonOperator(
        task_id='elbow_plot',
        python_callable=lambda ti: load_model_elbow(
            ti.xcom_pull(task_ids='preprocess_data')
        ),
    )

    t1 >> t2 >> [t3, t4]
