
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
import numpy as np

classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value)) 

attack = FastGradientMethod(estimator=classifier, eps=0.2)

x_test_adv = attack.generate(x=x_test)


predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Accuracy on adversarial samples: {accuracy*100}%")

