import json
import tensorflow
from utils import KerasDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from submission import SubmissionWriter
import os

""" 
    Example script demonstrating training on the SPEED+ dataset using Keras.
    Usage example: python keras_example.py --dataset [path to speed+] --epochs [num epochs] --batch [batch size] --run_gpu[True] --gpu_id [0]
"""


def evaluate(model, dataset, append_submission, dataset_root):

    """ Running evaluation on test set, appending results to a submission. """
    dataset_name = dataset
    if dataset=='test':
        dataset_name = 'synthetic'
        img_root = os.path.join(dataset_root, "synthetic","images")
        with open(os.path.join(dataset_root, "synthetic", dataset + '.json'), 'r') as f:
            image_list = json.load(f)
    else:
        img_root = os.path.join(dataset_root, dataset, "images")
        with open(os.path.join(dataset_root, dataset, 'test.json'), 'r') as f:
            image_list = json.load(f)

    print('Running evaluation on {} set...'.format(dataset))

    for img in image_list:
        img_path = os.path.join(img_root, img['filename'])
        pil_img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(pil_img)
        x = preprocess_input(x)
        x = np.expand_dims(x, 0)
        output = model.predict(x)
        append_submission(img['filename'], output[0, :4], output[0, 4:], dataset_name)

#======= Build Model ===========

def build_resnet(loss='mean_squared_error', optimizer='adam'):
    """ Initialized a ResNet50 architecture pre-trained on ImageNet """
    tensorflow.keras.backend.set_learning_phase(0)
    pretrained_model = tensorflow.keras.applications.ResNet50(weights="imagenet", include_top=False,
                                                              input_shape=(224, 224, 3))

    # Adding new trainable hidden and output layers to the model
    tensorflow.keras.backend.set_learning_phase(1)
    x = pretrained_model.output
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(1024, activation="relu")(x)
    predictions = tensorflow.keras.layers.Dense(7, activation="linear")(x)
    model_final = tensorflow.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
    model_final.compile(loss=loss, optimizer=optimizer)
    
    return model_final

#======= Build dataset ==========
def build_train_val_generator(speed_root, batch_size, fileextension='json'):
    
    """ Build a tensorflow data generator for training and validation data. """
    trainpath = os.path.join(speed_root, 'synthetic','train.'+fileextension)
    
    # Setting up parameters
    params = {'dim': (224, 224),
              'batch_size': batch_size,
              'n_channels': 3,
              'shuffle': True}
    
    with open(trainpath) as f:
        label_list = json.load(f)
    train_labels = label_list[:int(len(label_list)*.8)]
    validation_labels = label_list[int(len(label_list)*.8):]
    
    training_generator = KerasDataGenerator(preprocess_input, train_labels, speed_root, **params)
    validation_generator = KerasDataGenerator(preprocess_input, validation_labels, speed_root, **params)
    
    return training_generator, validation_generator

def main(speed_root, epochs, batch_size):

    """ Setting up data generators and model, training, and evaluating model on test and real_test sets. """
    
    # Loading and splitting dataset
    training_generator, validation_generator = build_train_val_generator(speed_root, batch_size)

    
    # Loading and freezing pre-trained model
    model_final = build_resnet(loss='mean_squared_error', optimizer='adam')

    # Training the model (transfer learning)
    history = model_final.fit(
        training_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[tensorflow.keras.callbacks.ProgbarLogger(count_mode='steps')])

    print('Training losses: ', history.history['loss'])
    print('Validation losses: ', history.history['val_loss'])
    
    
    # Generating submission
    submission = SubmissionWriter()
    evaluate(model_final, 'sunlamp', submission.append_real_test, speed_root)
    evaluate(model_final, 'lightbox', submission.append_real_test, speed_root)
    submission.export(suffix='keras_example')
    
    model_final.save("obj/KERAS_ALL_imagenet")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='speedplus/')
    parser.add_argument('--epochs', help='Number of epochs for training.', default=2)
    parser.add_argument('--batch', help='number of samples in a batch.', default=32)
    parser.add_argument('--run_gpu', help='Boolean to define the use or not of gpu', default=True)
    parser.add_argument('--gpu_id', help='Selection of the gpu to use', default=0)
    args = parser.parse_args()
    
    if args.run_gpu:
        os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    main(args.dataset, int(args.epochs), int(args.batch))