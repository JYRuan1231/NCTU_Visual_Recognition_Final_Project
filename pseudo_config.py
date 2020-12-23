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
# Training Config
##################################################


# model parameter
model_name = "efficientb4_v1"

batch_size = 16  # batch size
accumulation_steps = 1  # Gradient Accumulation
num_workers = 4  # number of Dataloader workers


# traning parameter
min_size = 256
max_size = 256
epochs = 1

# Scheduler parameter
learning_rate = 1e-3
momentum = 0.9
weight_decay = 5e-4
T_mult = 1
eta_min = 1e-5

##################################################
# pseudo label Testing Config
# ##################################################
pseudo_csv = "./data/pseudo_label.csv"
pseudo_model_name = 'pseudo_efficientb4_v1.pth'
result_pth = "./data/"
test_result = "pseudo_label.json"
