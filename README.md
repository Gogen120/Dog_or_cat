# Dog or cat

Neural network that predict, dog or cat is represented on the given image

# Quickstart

1. Download train data from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
2. Unzip to the current directory
3. Install all requirements `pip install -r requirements.txt # alternatively try pip3`
4. Run `python dogs_cats_nn.py` to train neural network. It will train neural network (it can be a rather long process)
4. Run the flask app `python app.py`. It will be available on localhost
5. If you want to retrain nn: make changes in `dogs_cats_nn.train_nn` function and run `python dogs_cats_nn.py`
