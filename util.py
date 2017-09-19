import cv2
from sys import platform

def plot_keras_history(history, plot_path, log_path, model_json_path, model):
    log_path = log_path.replace('detail.txt', 'acc{0:.2f}.txt'.format(history.history['acc'][-1]))
    with open(log_path, 'w') as f:
        for key in ['val_acc', 'acc', 'val_loss', 'loss']:
            try:
                f.write('\n{}='.format(key))
                f.write(str(history.history[key]))
            except Exception as e:
                pass


    # with open(model_json_path, 'w') as f:
        # f.write(model.to_json())

    if platform == "linux":
        return

    import matplotlib.pyplot as plt
    # summarize history for accuracy
    fig = plt.figure()
    fig.add_subplot(2,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    fig.add_subplot(2,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(plot_path)

def plot_image(image):
    if platform == "linux":
        return
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

def crop_skill_1(image, size):
    w = image.shape[0]
    h = image.shape[1]
    y1 = int(w * 385 / 480)
    y2 = int(w * 455 / 480)
    x1 = int(h * 600 / 848)
    x2 = int(h * 670 / 848)
    image = image[y1:y2, x1:x2]
    return cv2.resize(image, size)