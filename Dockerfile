FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --only-binary=:all: \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scikit-learn==1.4.2 \
        scipy==1.13.1 \
        xgboost==2.0.3 && \
    pip install \
        Flask==2.3.3 \
        gunicorn==21.2.0 \
        joblib==1.3.2 \
        Werkzeug==2.3.7 \
        click==8.1.7 \
        itsdangerous==2.1.2 \
        Jinja2==3.1.2 \
        MarkupSafe==2.1.3 \
        python-dateutil==2.9.0.post0 \
        pytz==2024.1 \
        six==1.16.0 \
        threadpoolctl==3.5.0 \
        requests==2.31.0 \
        urllib3==2.2.1 \
        certifi==2024.2.2 \
        charset-normalizer==3.3.2 \
        idna==3.7 \
        packaging==24.0 \
        tzdata==2024.1

COPY . .

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
