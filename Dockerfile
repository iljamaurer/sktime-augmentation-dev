FROM sktime-base

RUN pip install pytest pyod colorama
RUN pip uninstall sktime -y
ENV PYTHONPATH="/code/sktime:${PYTHONPATH}"