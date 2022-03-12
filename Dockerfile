FROM sktime-base

RUN pip install pytest pyod colorama pre-commit
RUN pip uninstall sktime -y
RUN pip install pre-commit
ENV PYTHONPATH="/code/sktime:${PYTHONPATH}"

RUN mkdir ~/pip
COPY docs_requirements.txt ~/pip/docs_requirements.txt
WORKDIR ~/pip
RUN pip install -r docs_requirements.txt
WORKDIR /code