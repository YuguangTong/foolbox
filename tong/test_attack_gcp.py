from foolbox.attacks import BoundaryAttack
from foolbox.models.googecloud import GoogleCloudModel
from foolbox.criteria import GoogleCloudTopKMisclassification, GoogleCloudTargetedClassScore
from keras.preprocessing import image
import numpy as np

# Load two images. The cat image is original image
# and the dog image is used to initialize a targeted
# attack.
dog_img = image.load_img('dog.jpg', target_size=(224, 224))
cat_img = image.load_img('cat.jpg', target_size=(224, 224))
dog_img = image.img_to_array(dog_img)
cat_img = image.img_to_array(cat_img)
cat_img = 2.0 * cat_img / 255.0 - 1
dog_img = 2.0 * dog_img / 255.0 - 1

dog_x = np.expand_dims(dog_img, axis=0)
cat_x = np.expand_dims(cat_img, axis=0)

# Build a foolbox model
gcp_model = GoogleCloudModel(bounds=[0, 255])

cat_label = 'cat'
dog_label = 'dog'

criterion_1 = GoogleCloudTargetedClassScore(dog_label, score=0.8)
criterion_2 = GoogleCloudTopKMisclassification(cat_label, k=5)
criterion = criterion_1 & criterion_2

attack = BoundaryAttack(model=gcp_model,
                        criterion=criterion)

iteration_size = 1000
global_iterations = 0
# Run boundary attack to generate an adversarial example
adversarial = attack(cat_img,
                     label=cat_label,
                     unpack=False,
                     iterations=iteration_size,
                     starting_point=dog_img,
                     log_every_n_steps=10,
                     verbose=True)