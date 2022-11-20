import os
import statistics
from keras.utils import load_img
from keras.utils import img_to_array
import tensorflow as tf
import torch
import glob
import torch.nn as nn
from skorch.helper import SliceDataset
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import pathlib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
from skorch import NeuralNetClassifier
from sklearn.model_selection import KFold, cross_validate, RepeatedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# Starting timer
start = timeit.default_timer()

# checking for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms by doing Data Augmentation to make sure all images have the same size
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

# Reproducability
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
train_loader = DataLoader(train_set, shuffle=True, batch_size=256)
test_loader = DataLoader(test_set, shuffle=True, batch_size=128)

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
    # plt.ylim(0.40000, 1)
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
    # plt.ylim(0.40000, 1)
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
    # plt.ylim(0.40000, 1)
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












