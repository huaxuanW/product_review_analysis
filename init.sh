conda deactivate
conda create -n nlp -y python=3.8
conda activate nlp
pip install -r requirements.txt
pip install torch==1.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html