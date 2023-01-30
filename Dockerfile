FROM python:3.8

RUN apt-get update -y && apt-get install libgl1-mesa-glx -y

WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install jinja2==3.0.0

CMD ["streamlit", "run", "app.py"]
# CMD ["bash"]