import os


ROOT_DIR = os.path.abspath(os.curdir)

CODE_DIR = os.path.join(ROOT_DIR, 'src')

DATA_DIR = os.path.join(ROOT_DIR, 'data')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')

MAX_LEN = 200

TOP_K = 50

DATA_TYPE = 'xlsx'

MODEL_PATH = os.path.join(CODE_DIR, 'model/0.92_max_len_200_saved_weights5.pkl')
MODEL_NAME = 'uer/roberta-base-finetuned-jd-full-chinese'

SAVED_PATH = os.path.join(OUTPUT_DIR, 'prediction.csv')