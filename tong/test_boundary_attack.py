import foolbox
from foolbox.models import KerasModel
from foolbox.attacks import BoundaryAttack
from foolbox.criteria import TargetClassProbability, TopKMisclassification, TargetClass
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import decode_predictions

# Load pretrained DenseNet
keras.backend.set_learning_phase(0)
kmodel = DenseNet121(weights='imagenet')

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
fmodel = KerasModel(kmodel, bounds=(-1, 1))

# label of the target class
preds = kmodel.predict(dog_x)
dog_label=np.argmax(preds)

# label of the original class
preds = kmodel.predict(cat_x)
cat_label=np.argmax(preds)

criterion_1 = TopKMisclassification(k=5)
criterion_2 = TargetClass(dog_label)
criterion_3 = TargetClassProbability(dog_label, p=0.5)
criterion = criterion_1 & criterion_2 & criterion_3

attack = BoundaryAttack(model=fmodel,
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
global_iterations += iteration_size

np.save('adversarial_image_{0}'.format(global_iterations), adversarial.image)

for i in range(10):
    adversarial = attack(adversarial,
                         unpack=False,
                         iterations=iteration_size,
                         verbose=True)
    global_iterations += iteration_size
    np.save('adversarial_image_{0}'.format(global_iterations), adversarial.image)

    # show results
    print(np.argmax(fmodel.predictions(adversarial.image)))
    print(fmodel.predictions(foolbox.utils.softmax(adversarial.image))[dog_label])
    preds = kmodel.predict(adversarial.image.copy())
    print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))