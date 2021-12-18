FROM sktime-base

RUN pip install pytest pyod colorama pre-commit
RUN pip uninstall sktime -y
RUN pre-commit install
ENV PYTHONPATH="/code/sktime:${PYTHONPATH}"