##################################################
# Config
##################################################

# generate testing/valid dataset

model_folder = "./models/"
train_path = "./data/train_images/"  # training dataset path
train_csv = "./data/train.csv"
test_path = "./data/test_images/"  # testing images
test_csv = "./data/test.csv"

##################################################
# Ensemble Config
##################################################


submission_csv = 'submission.csv'

# Ensemble learning model name
model_1 = "efficientb4_v1"
model_2 = "efficientb4_v2"
model_3 = "efficientb4_v3"
