#!/bin/bash

python enable_public_url.py

cd /content/easy-diffusion
pip install -r requirements.txt
pip install gradio
python app.py