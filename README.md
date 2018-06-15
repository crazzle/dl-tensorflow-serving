# Deep Learning Modelle deployen mit Keras und TensorFlow Serving

Der Beispielcode zum Artikel "Deep Learning Modelle deployen mit TensorFlow Serving"

## Requirements installieren

Das Training, der Client sowie auch TensorFlow-Serving nutzen verschiedene Python Pakete, von Numpy über TensorFlow bin hin zu Keras. Die Pakete können mit

```bash
pip install -r requirements.txt --user
```

installiert werden.

## Struktur

Im Ordner *code* befinden sich die Python Skripte für das Training der Modelle, sowie das Client-Script, dass mit
TensorFlow Serving spricht.

Im Ordner *notebooks* liegen 2 Expemplarische Jupyter Notebooks, die Training und Export inklusive Plots und Grafiken
darstellen.

## Skripte

*Wetterfrosch_V1_Training.py* und *V2* enthalten den Code für das Training der Modelle.
*Wetterfrosch_V1_Export.py* und *V2* enthalten den Code für die Modellexporte als TensorFlow-Graphen.

## TensorFlow Serving starten

Damit TensorFlow Serving gestartet werden kann wird Docker benötigt. Um den ModelServer zu starten wird zuerst das
Image gebaut:

```bash
docker build .
```

und anschließend der Container gestartet:
```bash
docker run -p 9000:9000 -v $(pwd)/models:/tmp/models -it tf_serving bash
```

Der Ordner *models* wird beim Start in den Container gemounted.