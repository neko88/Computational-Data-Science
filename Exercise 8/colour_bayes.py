import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.naive_bayes as skl
from skimage.color import lab2rgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import skimage as ski
import sys


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 113, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 186, 186),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=70, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)

def map_value(colour, features):
    result = features.loc[colour,:]
    label_val = result['value']
    return label_val

def main(infile):

    # Extract the colour data
    data = pd.read_csv(infile)

    # Normalize the colour data to 0 - 1 range
    data[['R', 'G', 'B']] = data[['R', 'G', 'B']] / 255

    # Split the data
    train, test = train_test_split(data, test_size=0.2)
    # Training data
    train_X = train[['R', 'G', 'B']] # array with shape (n, 3). Divide by 255 so components are all 0-1.
    train_y = train['Label'] # array with shape (n,) of colour words.
    # Validation data
    test_X = test[['R', 'G', 'B']]
    valid_y = test['Label']

    # Create naive Bayes classifer, train it
    model_rgb = skl.GaussianNB()
    model_rgb.fit(train_X.values, train_y.values)
    # Test the model & Evaluate the accuracy score & print
    score = model_rgb.score(test_X.values, valid_y.values)
    print(score)

    # Plot the results
    # (model_rgb)
    # plt.savefig('predictions_rgb.png')

    # Convert RGB values to LAB values
    # Pipeline ModeL
        # 1. Transformer: convert RGB -> LAB
        # 2. Apply Gaussian classifer

    # Create a Function Transformer to do colour-space conversion
    def rgb_to_lab(rgb_value):
        return ski.color.rgb2lab(rgb_value)

    transform_colour = FunctionTransformer(lambda x: rgb_to_lab(x), validate=True)

    model_lab = make_pipeline(
        transform_colour,
        skl.GaussianNB())

    # Train the model
    model_lab.fit(train_X, train_y)
    # Print the accuracy score.
    print(model_lab.score(test_X, valid_y))

    #plot_predictions(model_lab)
    #plt.savefig('predictions_lab.png')


if __name__ == '__main__':
    main(sys.argv[1])
