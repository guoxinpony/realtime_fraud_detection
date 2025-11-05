# build image 

FROM apache/airflow:2.10.5-python3.10

# transfer to root user 
USER root
COPY --chown=airflow:0 requirements.txt /opt/airflow/requirements.txt
COPY entrypoint.sh /bootstrap.sh
RUN chmod +x /bootstrap.sh
RUN mkdir -p /app/models

RUN apt-get update && apt-get install -y libgomp1

# Switch back to the airflow user and install the Python package in user space
USER airflow
# Debian/Bookworm's pip needs to allow "non-virtual environment installation"
RUN python -m pip install --no-cache-dir -r /opt/airflow/requirements.txt

ENTRYPOINT ["/bootstrap.sh"]