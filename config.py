import os.path
import torch


DATA_DIR = 'E:/data/vislab/landmarks_task/'
LOG_DIR = 'logs/seed'
N_EPOCHS = 20
BATCH_SIZE = 64
PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'

TEST_PATH_MENPO = os.path.join(DATA_DIR, 'Menpo/test')
TEST_PATH_300W = os.path.join(DATA_DIR, '300W/test')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')