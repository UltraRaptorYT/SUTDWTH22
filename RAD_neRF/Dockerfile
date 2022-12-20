FROM python:3.8.13

## App engine stuff
# Expose port you want your app on
EXPOSE 5000

# Upgrade pip 
RUN pip install -U pip

# Create a new directory for app (keep it in its own directory)
COPY . /app
WORKDIR app

RUN python -m pip install pyaudio

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

RUN bash scripts/install_ext.sh

# Run
ENTRYPOINT ["python", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]
