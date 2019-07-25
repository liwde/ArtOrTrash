# Ist das Kunst oder kann das weg?

Dieses Projekt will die uralte Frage ein für alle mal lösen: Ist das Kunst oder kann das weg? Und natürlich macht man das heutzutage mit Deep Machine Learning!

## FAQ

<dl>
  <dt>Wie funktioniert das eigentlich?</dt>
  <dd>Technisch gesehen mit Hilfe von Neuronalen Netzen und Deep Machine Learning. Praktisch gesehen mit einer Menge schwarzer Magie, Code-Snippets von Stack-Overflow und einem kleinen bisschen <a href="https://en.wikipedia.org/wiki/Confirmation_bias">Confirmation Bias</a>.</dd>
  <dt>Und wie gut klappt das?</dt>
  <dd>Großartig! Während der Entwicklung wurde das neuronale Netz nur auf insgesamt 600 Bildern trainiert. Die Alpha-Version basiert schon auf fast 6000 – sie ist also 10 mal so gut!</dd>
  <dt>Ist Kunst nicht subjektiv?</dt>
  <dd>Nein! Offensichtlich kann die App ganz klar vorhersagen, was Kunst ist und was weg kann. Das Problem ist also klar gelöst, Kunst ist ab jetzt rein objektiv bewertbar.</dd>
  <dt>Und das ist Wissenschaft?</dt>
  <dd><a href="https://www.youtube.com/watch?v=LSWDZsq0YoE&t=13m50s">Ja, Zu 95 Prozent!</a> Es muss wissenschaftlich absolut valide sein, denn dieses Projekt hat seine eigene DOI: <a href="https://doi.org/10.5281/zenodo.3349570">10.5281/zenodo.3349570</a></dd>
  <dt>Wie sieht es mit der Privatsphäre aus?</dt>
  <dd>Die App arbeitet lokal, es gibt keinen Server, keine Bilder werden irgendwohin gesendet oder gespeichert. Wenn du misstrauisch bist, kannst du sie auch einfach selbst kompilieren</dd>
  <dt>Wie kommt man auf so eine Idee?</dt>
  <dd>Gute Frage! Die Idee hatte ich tatsächlich schon länger. Auf der diesjährigen <a href="https://entropia.de/GPN19">Gulaschprogrammiernacht</a> des CCC Karlsruhe / Entropia e.V. hatte ich endlich mal Zeit, sie umzusetzen.</dd>
  <dt>Wie kriege ich die App!?</dt>
  <dd>Noch gibt es keine Version im PlayStore. Aber das Debug-APK und die Tensorflow/Keras-Modelle kann man unter <a href="https://github.com/liwde/ArtOrTrash/releases">Releases</a> finden.</dd>
</dl>

## Bestandteile

### `train_model`

Bestandteile des Projekts, um ein Tensorflow-Modell zu trainieren.

1. `./download_images.sh`: Bilder von Google Images herunterladen, die zum Training verwendet werden
2. Bilder manuell nach fehlerhaften JPEGs durchsuchen (das konnte ich nicht gescheit automatisieren)
3. `python3 ml.py`: Das eigentliche Machine Learning (Tensorboard: `tensorboard --logdir=logs`)
4. `python3 tflite.py`: Konvertiere das Keras-Modell nach TensorFlow Lite
5. `./copy_models.sh`: Kopiere das Modell in die Ressourcen der Android-App

**Voraussetzungen:** `requirements.txt`, ImageMagick, ein installierter ChromeDriver.

**Referenz:** Basiert im Wesentlichen auf [diesem Tutorial](https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13).

### `classifier_app`

Eine Android-App, die das Kamera-Bild live klassifiziert und sagt, ob es Kunst ist oder weg kann.

1. Die Schritte aus `train_model` ausführen.
2. Den Ordner in Android Studio öffnen und das Projekt bauen.

**Voraussetzungen:** Android Studio

**Referenz:** In weiten Teilen eine leicht modifizierte Variante der offiziellen [TensorFlow Lite Image Classification Example Application](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) ([Apache-2.0-Lizenz](https://www.apache.org/licenses/LICENSE-2.0.html))
