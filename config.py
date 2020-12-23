####### Training Config ############
# data
data_path = "./data/"
train_label_name = "train_labels.csv"
test_label_name = "test.csv"

# model
model_name = "efficientb4_v1.pth"
# training parameters
validation_percentage = 0.1
batch_size = 16
accumulation_steps = 1
num_workers = 4

optimizer_lr = 1e-3
optimizer_momentum = 0.9
optimizer_weight_decay = 5e-4

scheduler_exp_t0 = 1
scheduler_exp_t_mult = 1
scheduler_exp_eta_min = 1e-5

num_epochs = 50


# result
result_path = "./models/"
