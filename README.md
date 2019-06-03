# Ist das Kunst oder kann das weg?

Dieses Projekt will die uralte Frage ein für alle mal lösen: Ist das Kunst oder kann das weg? Natürlich macht man das heutzutage mit Deep Machine Learning.

## `train_model`

Bestandteile des Projekts, um ein Tensorflow-Modell zu trainieren.

1. `./download_images.sh`: Bilder von Google Images herunterladen, die zum Training verwendet werden
2. Bilder manuell nach Fehlerhaf
3. `python3 ml.py`: Das eigentliche Machine Learning (Tensorboard: `tensorboard --logdir=logs`)
4. `python3 tflite.py`: Konvertiere das Keras-Modell nach TensorFlow Lite
5. `./copy_models.sh`: Kopiere das Modell in die Ressourcen der Android-App

**Voraussetzungen:** `requirements.txt`, ImageMagick, ein installierter ChromeDriver.

**Referenz:** Basiert im Wesentlichen auf [diesem Tutorial](https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13).

## `classifier_app`

Eine Android-App, die das Kamera-Bild live klassifiziert und sagt, ob es Kunst ist oder weg kann.

1. Die Schritte aus `train_model` ausführen.
2. Den Ordner in Android Studio öffnen und das Projekt bauen.

**Voraussetzungen:** Android Studio

**Referenz:** In weiten eine leicht modifizierte Variante der offiziellen [TensorFlow Lite Image Classification Example Application](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) ([Apache-2.0-Lizenz](https://www.apache.org/licenses/LICENSE-2.0.html))
