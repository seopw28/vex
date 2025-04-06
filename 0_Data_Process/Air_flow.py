from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.hive_operator import HiveOperator
from airflow.sensors.hive_partition_sensor import HivePartitionSensor
from datetime import datetime, timedelta

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'basic_example_dag',
    default_args=default_args,
    description='A basic example DAG with Hive operator and sensor',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Define a Python function to be executed by the task
def print_hello():
    return 'Hello from Airflow!'

# Create a task using the Python function
hello_task = PythonOperator(
    task_id='hello_task',
    python_callable=print_hello,
    dag=dag,
)

# Create a Hive sensor task to check if a partition exists
hive_sensor = HivePartitionSensor(
    task_id='hive_sensor',
    table='your_database.your_table',
    partition="dt='{{ ds }}'",  # Uses the execution date as partition
    mode='reschedule',
    poke_interval=300,  # Check every 5 minutes
    timeout=3600,  # Timeout after 1 hour
    dag=dag,
)

# Create a Hive operator task to run a Hive query
hive_query = HiveOperator(
    task_id='hive_query',
    hql="""
    SELECT 
        COUNT(*) as count,
        SUM(amount) as total_amount
    FROM your_database.your_table
    WHERE dt = '{{ ds }}'
    """,
    hive_cli_conn_id='hive_default',
    dag=dag,
)

# Set task dependencies
hello_task >> hive_sensor >> hive_query 
