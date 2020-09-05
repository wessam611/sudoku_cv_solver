TRAIN_FOLDER = '../train'
TRAIN_IMGS_PATH = '{}/images'.format(TRAIN_FOLDER)
TRAIN_CSV = '{}/labels.csv'.format(TRAIN_FOLDER)


TEST_FOLDER = '../test'
TEST_IMGS_PATH = '{}/images'.format(TEST_FOLDER)

IMAGE_SIZE = [256, 256]
BOX_SIZE = [28, 28]

SAVED_MODEL = 'saved_model/model'

BATCH_SIZE = 512
EPOCHS = 1000

TRAIN_DATA_DIR = 'train_digits'
SUBMISSION_CSV = 'submission.csv'