# ChatGPT, DALL·E & Voice

### Cloning the Repository

    git clone https://github.com/evadelzz1/experiment2.git

### Setting up a Virtual Environment

    cd ./experiment2

    pyenv versions

    pyenv local 3.11.6

    pyenv versions

    echo '.env'  >> .gitignore
    echo '.venv' >> .gitignore

    echo 'OPENAI_API_KEY=sk-9jz....' >> .env
    echo 'USER_PASSWORD=password'    >> .env

    ls -la

### Activate the virtual environment

    python3 -m venv .venv

    source .venv/bin/activate

    python -V

### Install the required dependencies

    pip list
    
    pip install -r requirements.txt
    
    pip freeze | tee requirements.txt.detail

### Running the Application

    python -m streamlit run 1_Home.py

### Deactivate the virtual environment

    deactivate

.