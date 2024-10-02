from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mnist = fetch_openml("mnist_784")

img1 = np.array(mnist.data)[0]
#print(np.array(mnist.target)[0])
#img1 = img1.reshape(28, 28)
#plt.imshow(img1, cmap='gray')
#plt.show()

scaler =  StandardScaler()
X = scaler.fit_transform(mnist.data)

X_train, X_test, Y_train, y_test = train_test_split(X, mnist.target, test_size=0.2, random_state=42 )

model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

print(model.predict([img1]))





