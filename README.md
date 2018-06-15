# Big Data mit Luigi und Python
Der Beispielcode zum Artikel "Deep Learning Modelle deployen mit TensorFlow Serving"

## Requirements installieren
Das Training, der Client sowie auch TensorFlow-Serving nutzen verschiedene Python Pakete, von Numpy über TensorFlow bin hin zu Keras. Die Pakete können mit

```bash
pip install -r requirements.txt --user
```

installiert werden.

## Training
*Wetterfrosch_V1.py* enthält den Code für das Training sowie den Export der ersten Version des Wetterfroschs.

Der Code läd das CSV und vorverarbeitet die Daten, damit das Netz besser konvergiert.

*Download* und *Clean* nutzen Standard Python Libraries (Pandas, PRAW, NLTK). Der *Training* Task ist als PySpark-Job implementiert.

Gestartet wird die Pipeline mit

```bash
PYTHONPATH='.' luigi --module 00_training_pipeline TrainModel --version 1 \
                                                              --local-scheduler
```

## Klassifikation
*01_classification_pipeline.py* enhtält den Code für die tägliche Klassifikationspipeline.

*Fetch* und *Clean* nutzen Standard Python Libraries (Pandas, PRAW, NLTK). Der *Classify* Task ist als PySpark-Job implementiert.

Gestartet wird die Pipeline mit

```bash
PYTHONPATH='.' luigi --module 01_classification_pipeline RangeDailyBase --of Classify \
                                                                        --stop=$(date +"%Y-%m-%d") \
                                                                        --days-back 4 \
                                                                        --Classify-version 1 \
                                                                        --reverse \
                                                                        --local-scheduler
```
