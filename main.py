import glob
import os
import pathlib
import statistics
import timeit
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from keras.utils import img_to_array
from keras.utils import load_img
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_validate
from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

warnings.filterwarnings('ignore')

# Starting timer
start = timeit.default_timer()

# checking for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms by doing Data pre-processing to make sure all images have the same size
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                         [0.5, 0.5, 0.5])
])

image_categories = os.listdir("Vegetables Image Classification/Train_Test_Dataset")
path_for_images_example = "Vegetables Image Classification/Train_Test_Dataset"


def plot_images(image_categories):
    # Create a figure
    plt.figure(figsize=(12, 12))
    for i, cat in enumerate(image_categories):
        # Load images for the ith category
        image_path = path_for_images_example + '/' + cat
        images_in_folder = os.listdir(image_path)
        first_image_of_folder = images_in_folder[0]
        first_image_path = image_path + '/' + first_image_of_folder
        img = load_img(first_image_path)
        img_arr = img_to_array(img) / 255.0

        # Create Subplot and plot the images
        plt.subplot(4, 4, i + 1)
        plt.imshow(img_arr)
        plt.title(cat)
        plt.axis('off')

    plt.show()


# Call the function
plot_images(image_categories)


# Reproducibility
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


set_seed()

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='Large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
plt.rc('xtick', labelsize=6)

# Path for training and testing directory
entireDataset = os.path.join("Vegetables Image Classification/Train_Test_Dataset")
train_test_set = torchvision.datasets.ImageFolder(entireDataset, transform=transformer)

train_set_size = int(len(train_test_set) * 0.75)
test_set_size = len(train_test_set) - train_set_size
train_set, test_set = data.random_split(train_test_set, [train_set_size, test_set_size])
# Prediction Train Model
y_train = np.array([labels for images, labels in iter(train_set)])

# Dataloader
train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
test_loader = DataLoader(test_set, shuffle=True, batch_size=16)

# Storing the different number of images for each class
_, _, Bean_files = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Bean"))
Bean_count = len(Bean_files)
_, _, Bitter_Gourd_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Bitter_Gourd"))
Bitter_Gourd_count = len(Bitter_Gourd_file)
_, _, Bottle_Gourd_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Bottle_Gourd"))
Bottle_Gourd_count = len(Bottle_Gourd_file)
_, _, Brinjal_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Brinjal"))
Brinjal_count = len(Brinjal_file)
_, _, Broccoli_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Broccoli"))
Broccoli_count = len(Broccoli_file)
_, _, Cabbage_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Cabbage"))
Cabbage_count = len(Cabbage_file)
_, _, Capsicum_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Capsicum"))
Capsicum_count = len(Capsicum_file)
_, _, Carrot_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Carrot"))
Carrot_count = len(Carrot_file)
_, _, Cauliflower_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Cauliflower"))
Cauliflower_count = len(Cauliflower_file)
_, _, Cucumber_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Cucumber"))
Cucumber_count = len(Cucumber_file)
_, _, Papaya_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Papaya"))
Papaya_count = len(Papaya_file)
_, _, Potato_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Potato"))
Potato_count = len(Potato_file)
_, _, Pumpkin_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Pumpkin"))
Pumpkin_count = len(Pumpkin_file)
_, _, Radish_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Radish"))
Radish_count = len(Radish_file)
_, _, Tomato_file = next(os.walk("Vegetables Image Classification/Train_Test_Dataset/Tomato"))
Tomato_count = len(Tomato_file)

# categories
root = pathlib.Path(entireDataset)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

print("\nClasses: ", classes)

# Plotting the graph showing the different classes
classes_types = ['Bean', 'Bitter\nGourd', 'Bottle\nGourd', 'Brinjal', 'Broccoli',
                 'Cabbage', 'Caps-\nicum', 'Carrot', 'Cauli-\nflower', 'Cucu-\nmber',
                 'Papaya', 'Potato', 'Pumpkin', 'Radish']
classes_distribution = [Bean_count, Bitter_Gourd_count, Bottle_Gourd_count, Brinjal_count,
                        Broccoli_count, Cabbage_count, Capsicum_count, Carrot_count, Cauliflower_count,
                        Cucumber_count, Papaya_count, Potato_count, Pumpkin_count, Radish_count]
plt.bar(classes_types, classes_distribution)
plt.ylabel("Number of images")
plt.xlabel("Classes")
plt.title("Classes distribution")
plt.savefig("Graphs/Classes Distribution.png")
plt.show()
plt.clf()


# Bar Chart plotting for training and validation data for all folds at the same time
def plot_result_all_folds(x_label, y_label, plot_title, train_acc, test_acc, train_prec, test_prec,
                          train_rec, test_rec, train_f1, test_f1, number_folds):
    plt.figure(figsize=(12, 6))
    axis_labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold", "7th Fold", "8th Fold",
                   "9th Fold"
        , "10th Fold"]
    X_axis = np.arange(number_folds)
    ax = plt.gca()
    plt.bar(X_axis - 0.4, train_acc, 0.1, color='#1C2DAD', label='Train_accuracy')
    plt.bar(X_axis - 0.3, test_acc, 0.1, color='#1566AD', label='Test_accuracy')
    plt.bar(X_axis - 0.2, train_prec, 0.1, color='#AD232C', label='Train_precision')
    plt.bar(X_axis - 0.1, test_prec, 0.1, color='#AD238D', label='Test_precision')
    plt.bar(X_axis, train_rec, 0.1, color='#2BAD1A', label='Train_recall')
    plt.bar(X_axis + 0.1, test_rec, 0.1, color='#79BC6A', label='Test_recall')
    plt.bar(X_axis + 0.2, train_f1, 0.1, color='#A7A50C', label='Train_F1')
    plt.bar(X_axis + 0.3, test_f1, 0.1, color='#E4E415', label='Test_F1')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, axis_labels[0:number_folds])
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig("Graphs/Summary per fold.png")
    plt.clf()


# Bar Chart plotting Precision, Recall, F-1 measure and accuracy for one fold at a time
def plot_result_each_fold(x_label, y_label, plot_title, tst_precision, tst_recall, tst_f1, tst_accuracy,
                          tr_precision, tr_recall, tr_f1, tr_accuracy, current_fold):
    plt.figure(figsize=(12, 6))
    axis_labels = ["Precision", "Recall", "F1-Measure", "Accuracy"]
    X_axis = np.arange(len(axis_labels))
    test = [tst_precision, tst_recall, tst_f1, tst_accuracy]
    train = [tr_precision, tr_recall, tr_f1, tr_accuracy]
    ax = plt.gca()
    plt.bar(X_axis - 0.2, train, 0.4, color='blue', label="Train")
    plt.bar(X_axis + 0.2, test, 0.4, color='red', label="Validation")
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, axis_labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    fig_title = "Graphs/Scores for Fold #" + str(current_fold) + ".png"
    plt.savefig(fig_title)
    plt.clf()


# Bar Chart plotting for training and validation data aggregated for all folds
def plot_result_aggregate(x_label, y_label, plot_title, tst_precision, tst_recall, tst_f1, tst_accuracy,
                          tr_precision, tr_recall, tr_f1, tr_accuracy, number_folds):
    plt.figure(figsize=(12, 6))
    axis_labels = ["Precision", "Recall", "F1-Measure", "Accuracy"]
    X_axis = np.arange(len(axis_labels))
    test = [tst_precision, tst_recall, tst_f1, tst_accuracy]
    train = [tr_precision, tr_recall, tr_f1, tr_accuracy]
    ax = plt.gca()
    plt.bar(X_axis - 0.2, train, 0.4, color='orange', label="Train")
    plt.bar(X_axis + 0.2, test, 0.4, color='green', label="Validation")
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, axis_labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    fig_title = "Graphs/Aggregate Scores for " + str(number_folds) + " folds.png"
    plt.savefig(fig_title)
    plt.clf()


class CNN(nn.Module):
    def __init__(self, num_classes=14):
        super(CNN, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # Input shape= (256,3,150,150)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        # Shape= (256,6,150,150)
        self.bn1 = nn.BatchNorm2d(6)
        # Shape= (256,6,150,150)
        self.relu1 = nn.ReLU()
        # Shape= (256,6,150,150)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (256,6,75,75)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (256,12,75,75)
        self.relu2 = nn.ReLU()
        # Shape= (256,12,75,75)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        # Shape= (256,24,75,75)
        self.bn3 = nn.BatchNorm2d(24)
        # Shape= (256,24,75,75)
        self.relu3 = nn.ReLU()
        # Shape= (256,24,75,75)

        self.fc = nn.Linear(in_features=75 * 75 * 24, out_features=num_classes)

        # Feed forward function

    def forward(self, x):
        # conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Above output will be in matrix form, with shape (256,32,75,75)
        # flatten
        x = x.view(-1, 24 * 75 * 75)
        # fc layer
        x = self.fc(x)

        return x


model = CNN(num_classes=15).to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

# Backprop and optimisation
optimizer.zero_grad()
optimizer.step()

# calculating the size of training and testing images
train_count = len(train_set + '/**/*')
test_count = len(test_set + '/**/*')
entireDataset_count = len(glob.glob(entireDataset + '/**/*'))
print("\nNumber of entire Dataset: ", entireDataset_count, " Images")
print("Number of Train Images: ", train_count, ", Number of Test Images: ", test_count, "\n")

# Model training and saving best model
best_accuracy = 0.0

total_step = len(train_loader)
best_actual_list = []
best_prediciton_list = []
loss_list = []
acc_list = []

# Hyper parameters to the model
num_epochs = 2
learningRate = 0.001
numberOfFolds = 10
weight_decay = 0.0001
batch_size = 64

torch.manual_seed(0)

net = NeuralNetClassifier(
    model,
    max_epochs=num_epochs,
    lr=learningRate,
    batch_size=batch_size,
    optimizer=optim.Adam,
    criterion=criterion,
    device=device
)
print("// -------------------------------  Testing the tensor size, shape, and data matrix on the Model "
      "before training it.  ------------------------------- //")
print()
net.fit(train_set, y=y_train)
print()

print("// -------------------------------  Training the Model with " + str(numberOfFolds) + " Folds, using "
      + str(num_epochs), "Epochs ------------------------------- //")
print()

model.train()

# Define the Score Metrics
scoring = {'accuracy': make_scorer(accuracy_score, normalize=True),
           'precision': make_scorer(precision_score, average='weighted'),
           'recall': make_scorer(recall_score, average='weighted'),
           'f1_score': make_scorer(f1_score, average="weighted")}

# K-fold iterator to be passed to the cross validation function
kf = KFold(n_splits=numberOfFolds, shuffle=True, random_state=0)

# Passing the scores metric dictionary and the KFold iterator to the cross_validate training function
train_sliceable = SliceDataset(train_set)
scores = cross_validate(net, train_sliceable, y_train, cv=kf, scoring=scoring, error_score="raise",
                        return_train_score=True)

# All the required metrics
# print()
# print(scores.keys())
print()
print("// -----------------------------------  ", 'Train Accuracy: ' + str(scores['train_accuracy'][9]),
      " -----------------------------------  //")
print()
print("// -----------------------------------  ", 'Test Accuracy: ' + str(scores['test_accuracy'][9]),
      " -----------------------------------  //")
print()
# print(scores['train_accuracy'])
# print()
# print(scores['test_accuracy'])
print("// -----------------------------------  ", 'The remaining metrics of for the training and the test',
      " -----------------------------------  //")
print(scores)
print()

# Looping through the dictionary to produce a graph per fold
n = 0
for i in range(numberOfFolds):
    n = n + 1
    title_graph = "Train & Validation scores for fold#" + str(n)
    plot_result_each_fold("Scores",
                          "Value",
                          title_graph,
                          scores["test_precision"][n - 1],
                          scores["test_recall"][n - 1],
                          scores["test_f1_score"][n - 1],
                          scores["test_accuracy"][n - 1],
                          scores["train_precision"][n - 1],
                          scores["train_recall"][n - 1],
                          scores["train_f1_score"][n - 1],
                          scores["train_accuracy"][n - 1],
                          n)

# Looping through the dictionary to produce an aggregate
n = 0
agg_tr_prec = []
agg_tr_rec = []
agg_tr_acc = []
agg_tr_f1 = []
agg_tst_prec = []
agg_tst_rec = []
agg_tst_acc = []
agg_tst_f1 = []

for i in range(numberOfFolds):
    n = n + 1
    agg_tr_prec.append(scores["train_precision"][n - 1])
    agg_tr_rec.append(scores["train_recall"][n - 1])
    agg_tr_acc.append(scores["train_accuracy"][n - 1])
    agg_tr_f1.append(scores["train_f1_score"][n - 1])
    agg_tst_prec.append(scores["test_precision"][n - 1])
    agg_tst_rec.append(scores["test_recall"][n - 1])
    agg_tst_acc.append(scores["test_accuracy"][n - 1])
    agg_tst_f1.append(scores["test_f1_score"][n - 1])

title_graph = "Aggregate Train & Validation scores for " + str(numberOfFolds) + " folds"
plot_result_aggregate("Scores",
                      "Value",
                      title_graph,
                      statistics.mean(agg_tst_prec),
                      statistics.mean(agg_tst_rec),
                      statistics.mean(agg_tst_f1),
                      statistics.mean(agg_tst_acc),
                      statistics.mean(agg_tr_prec),
                      statistics.mean(agg_tr_rec),
                      statistics.mean(agg_tr_f1),
                      statistics.mean(agg_tr_acc),
                      numberOfFolds)

# Displaying overall folds performance
title_graph = "Scores comparison across" + str(numberOfFolds) + " folds"
plot_result_all_folds("Scores",
                      "Value",
                      title_graph,
                      scores["train_accuracy"],
                      scores["test_accuracy"],
                      scores["train_precision"],
                      scores["test_precision"],
                      scores["train_recall"],
                      scores["test_recall"],
                      scores["train_f1_score"],
                      scores["test_f1_score"],
                      numberOfFolds)

plt.figure(figsize=(18, 18))
plot_confusion_matrix(net, train_set, y_train.reshape(-1, 1))
plt.xlabel('Predicted Values', fontsize=10)
plt.ylabel('Actual Values', fontsize=10)
labels = ['Bean', 'Bitter\nGourd', 'Bottle\nGourd', 'Brinjal', 'Broccoli',
          'Cabbage', 'Caps-\nicum', 'Carrot', 'Cauli-\nflower', 'Cucu-\nmber',
          'Papaya', 'Potato', 'Pumpkin', 'Radish']
X_axis = np.arange(len(labels))

plt.xticks(X_axis, labels)
plt.rc('xtick', labelsize=6)

plt.savefig("Graphs/Confusion Matrix (Trained model)")
plt.clf()
print()

for epoch in range(num_epochs):
    print("// -----------------------------------  ", 'Epoch: ' + str(epoch + 1),
          " for Testing Evaluation -----------------------------------  //")

    model.eval()
    test_accuracy = 0.0

    actual_data = []
    prediction_data = []

    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

        # appending the data iterated on the current batch with the rest of the data
        actual_data.append(labels.data.tolist())
        prediction_data.append(prediction.tolist())

    test_accuracy = test_accuracy / test_count

    # formatting actual_data and prediction_data to remove inner lists [[4,5], [9]] => [4,5,9]
    flat_list_actual = [x for xs in actual_data for x in xs]
    flat_list_prediction = [x for xs in prediction_data for x in xs]

    # Confusion Matrix
    cm = confusion_matrix(flat_list_actual, flat_list_prediction)

    # Classification Report
    cr = classification_report(flat_list_actual, flat_list_prediction, labels=np.unique(flat_list_prediction))

    print()
    print("Classification report: ")
    print(cr)
    print()
    print()

    # Plot confusion_matrix
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    ax.set_title('Confusion Matrix for the different classifiers\n\n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    try:
        ax.xaxis.set_ticklabels(['Bean', 'Bitter\nGourd', 'Bottle\nGourd', 'Brinjal', 'Broccoli',
                                 'Cabbage', 'Caps-\nicum', 'Carrot', 'Cauli-\nflower', 'Cucu-\nmber',
                                 'Papaya', 'Potato', 'Pumpkin', 'Radish'])
        ax.yaxis.set_ticklabels(['Bean', 'Bitter\nGourd', 'Bottle\nGourd', 'Brinjal', 'Broccoli',
                                 'Cabbage', 'Caps-\nicum', 'Carrot', 'Cauli-\nflower', 'Cucu-\nmber',
                                 'Papaya', 'Potato', 'Pumpkin', 'Radish'])
    except:
        print()

    plt.savefig("Graphs/Confusion Matrix (Epoch# " + str(epoch + 1) + ")", bbox_inches='tight')
    plt.clf()

    # Saving the best model so far, by comparing it with the test's accuracy from the previous models.
    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'trainingmodel.model')
        best_accuracy = test_accuracy
        best_actual_list = flat_list_actual
        best_prediciton_list = flat_list_prediction

# Displaying the best model
print("\n// -------------------------------------  The best model data",
      " -------------------------------------  //")
print("Test Accuracy: ", best_accuracy)
bcm = confusion_matrix(best_actual_list, best_prediciton_list)
bcr = classification_report(best_actual_list, best_prediciton_list, labels=np.unique(best_prediciton_list))
ax = sns.heatmap(bcm, annot=True, cmap='Blues', fmt='g')
ax.set_title('Confusion Matrix for the best model\n\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
try:
    ax.xaxis.set_ticklabels('Bean', 'Bitter\nGourd', 'Bottle\nGourd', 'Brinjal', 'Broccoli',
                            'Cabbage', 'Caps-\nicum', 'Carrot', 'Cauli-\nflower', 'Cucu-\nmber',
                            'Papaya', 'Potato', 'Pumpkin', 'Radish')
    ax.yaxis.set_ticklabels('Bean', 'Bitter\nGourd', 'Bottle\nGourd', 'Brinjal', 'Broccoli',
                            'Cabbage', 'Caps-\nicum', 'Carrot', 'Cauli-\nflower', 'Cucu-\nmber',
                            'Papaya', 'Potato', 'Pumpkin', 'Radish')
except:
    print()

plt.savefig("Graphs/Best Model Confusion Matrix", bbox_inches='tight')
plt.clf()
print("Classification Matrix:\n", bcr)

# Stopping and Displaying timer
stop = timeit.default_timer()
print()
print("\n*** Program finished executing! ***")
print("Program Run Time for ", num_epochs, " epochs: ", "{:.2f}".format((stop - start) / 60), " minutes")
