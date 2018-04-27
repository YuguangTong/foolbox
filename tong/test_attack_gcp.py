from foolbox.attacks import BoundaryAttack
from foolbox.models import GoogleCloudModel
from foolbox.criteria import GoogleCloudTopKMisclassification, GoogleCloudTargetedClassScore
from keras.preprocessing import image
import numpy as np
import csv

def get_label_set(label_filename):
    label_file = open(label_filename,'r')
    label_rows = csv.reader(label_file)
    label_list = []
    for row in label_rows:
        for item in row:
            label_list.append(item.lower().strip())
    return set(label_list)

cat_label_filename = 'cat_labels.txt'
dog_label_filename = 'dog_labels.txt'
cat_labels_set = get_label_set(cat_label_filename)
dog_labels_set = get_label_set(dog_label_filename)

# Load two images. The cat image is original image
# and the dog image is used to initialize a targeted
# attack.
dog_img = image.load_img('dog.jpg', target_size=(224, 224))
cat_img = image.load_img('cat.jpg', target_size=(224, 224))
dog_img = image.img_to_array(dog_img)
cat_img = image.img_to_array(cat_img)

dog_x = np.expand_dims(dog_img, axis=0)
cat_x = np.expand_dims(cat_img, axis=0)

# Build a foolbox model
gcp_model = GoogleCloudModel(bounds=[0, 255])

cat_label = 'cat'
dog_label = 'dog'

criterion_1 = GoogleCloudTargetedClassScore(dog_label,
                                            score=0.8,
                                            target_class_lookup_table=dog_labels_set)
criterion_2 = GoogleCloudTopKMisclassification(cat_label,
                                               k=5,
                                               original_class_lookup_table=cat_labels_set)
criterion = criterion_1 & criterion_2

attack = BoundaryAttack(model=gcp_model,
                        criterion=criterion)

iteration_size = 100
global_iterations = 0
# Run boundary attack to generate an adversarial example
adversarial = attack(cat_img,
                     label=cat_label,
                     unpack=False,
                     iterations=iteration_size,
                     starting_point=dog_img,
                     log_every_n_steps=10,
                     verbose=True)
