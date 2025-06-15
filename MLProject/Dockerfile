FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda env create -f conda.yaml

CMD ["conda", "run", "--no-capture-output", "-n", "mlflow-env", "python", "modelling.py", "--data_path", "dataset_preprocessing.zip"]
