FROM gmxapi/ci-mpich

RUN . $HOME/testing/bin/activate && \
    pip install -r /home/testing/gmxapi/requirements-docs.txt
COPY documentation /home/testing/gmxapi/documentation
RUN cd /home/testing/gmxapi && \
    . $HOME/testing/bin/activate && \
    sphinx-build -b html documentation html

