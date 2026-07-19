FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir monai nibabel numpy
CMD ["python3", "src/evaluate_spleen.py"]
