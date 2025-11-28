# Use official Python runtime
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}
COPY requirements.txt .
# Install system dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8080

CMD ["main.lambda_handler"]
