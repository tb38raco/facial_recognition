#       __Kurzanleitung Gesichtserkennung & -verifikation__
## 0    Verwendete Module

__DeepFace__: DeepFace ist eine Bibliothek für Python, die sich auf Gesichtsverarbeitung und 
-erkennung spezialisiert hat. Sie ermöglicht die Arbeit mit Gesichtsmerkmalen, Gesichtsverifikation
(Überprüfung, ob zwei Gesichter derselben Person gehören) und Emotionserkennung in Bildern.

__OpenCV__ (cv2): OpenCV ist eine Open-Source-Bibliothek für Computer Vision. Diese wird verwendet, 
um Bilder und Videos zu verarbeiten, Gesichtserkennung durchzuführen und visuelle Aufgaben in 
Python zu automatisieren.

__FER__ (Facial Expression Recognition): FER ist eine Bibliothek für die Erkennung von Gesichtsausdrücken.
Sie hilft dabei, Emotionen innerhalb von Gesichtern zu erkennen, wie zum Beispiel glücklich oder traurig.

__Tabulate__: Die tabulate-Bibliothek ist nützlich, um Tabellen mit Daten zu erstellen und sie in einer 
übersichtlichen Form darzustellen. Im gegebenen Code wird sie verwendet, um Ergebnisse in Tabellenform
auszugeben.

Die Bibliotheken lassen sich via 'pip install *Bibliothekname*' installieren
```python
import os
from deepface import DeepFace
import cv2
from fer import FER
from tabulate import tabulate
```

## 1      Automatisierte Gesichtserkennung und -verifikation

### 1.1   Vergleichsbilder

Zuerst wählen wir einen Ordner aus, der Bilder mit Gesichtern enthählt. Diser sollte sich im selben 
Verzeichnis wie unser Code befinden
```python
# Pfad zum Ordner mit den Bildern
folder_path = 'IMG_1001.JPG'
```

### 1.2 Erkennungs- & Verifikationsprozess  
In diesem Code werden viele Schritte durchgeführt, um Gesichter in Bildern zu erkennen und dann festzustellen, wie ähnlich oder verschieden diese Gesichter sind.
Mit der for-Schleife betrachten wir jedes einzelne Bild im Ordner.
        Dann nehmen wir zwei Bilder und erstellen ein "Bildpaar" aus diesen beiden Bildern, um zu überprüfen, wie ähnlich oder verschieden sie sind.
        Erkenne Gesichter: Wir schauen uns jedes Bild in diesem Bildpaar an und versuchen, Gesichter darauf zu finden.
        Durchführen der Gesichtsverifikation: Wenn wir in beiden Bildern Gesichter finden, führen wir eine *Gesichtsverifikation* durch.
        Für diese Überprüfung verwenden ein Modell namens "VGG-Face".
        [VGG-Face](https://exposing.ai/vgg_face/) ist ein tiefes [neuronales Netzwerkmodell](https://www.heise.de/ratgeber/Neuronale-Netz-einfach-erklaert-6343697.html), dessen Hauptaufgabe in der Gesichtserkennung und der Extraktion von Gesichtsmerkmalen besteht.
        Dazu wird jedes Bild zunächst durch Vorverarbeitung auf ein einheitliches Format gebracht und anschließend aus jedem erkannten Gesicht ein sogenannter *Merkmalsvektor* erzeugt. Diese Übersetzung eines Bildes in eine für Maschinen erfassbare Sprache, also die der Mathematik, bildet die Grundlage für alle weiteren Aufgaben, die unser Gesichtserkennungssystem erfüllen soll. Diese Vorgehensweise ist typisch für Deep Learning-Bildverarbeitungsmodelle.
        Die Deepface-Bibliothek erlaubt außerdem eine Auswahl unterschiedlichster Modell und Metriken. So können wir beispielsweise in Zeile 18 'VGG-Face' durch 'OpenFace' ersetzen und die Ergebnisse in Abhängigkeit des verwendeten Modells beobachten. Genauere Informationen über verfügbare Alternativen findet ihr [hier](https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py).
```python

# Liste, um die Ergebnisse der Gesichtsverifikation zu speichern
results = []

# Durchlaufe alle Bilder im Ordner
files = os.listdir(folder_path)
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        # Pfade zu den Bildern erstellen
        img1_path = os.path.join(folder_path, files[i])
        img2_path = os.path.join(folder_path, files[j])

        # Gesichter in den Bildern erkennen (mit DeepFace)
        detected_faces_img1 = DeepFace.extract_faces(img1_path, enforce_detection=False)
        detected_faces_img2 = DeepFace.extract_faces(img2_path, enforce_detection=False)

        if detected_faces_img1 is not None and detected_faces_img2 is not None:
            # Verifikationsprozess durchführen
            result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name='VGG-Face', enforce_detection=False)

            # Ergebnis zur Ergebnisliste hinzufügen (als Liste, nicht als Tupel)
            results.append([files[i], files[j], result["distance"], result["verified"]])
        else:
            # Wenn Gesichter nicht erkannt wurden, geben Sie den Bildtitel aus
            print(f"Gesichter in {files[i]} und {files[j]} wurden nicht erkannt.")
```

### 1.3 Formatierung der Ergebnisse
```python
# Erstellen Sie eine Tabelle mit den Ergebnissen
table_headers = ["Bild 1", "Bild 2", "Distanz", "Wahrheitswert"]
table_data = results

# Tabelle anzeigen
print(tabulate(table_data, headers=table_headers, tablefmt="pretty"))
```

Der code wird so verändert, dass das Ergebnis in Tabellenform angezeigt wird.
```python
# Ergebnisse ausgeben
print("Bild 1\t\tBild 2\t\tDistanz\t\tWahrheitswert")
print("------------------------------------------------------------")
for filename1, filename2, distance, verified in results:
    print(f"{filename1}\t\t{filename2}\t\t{distance}\t\t{verified}")

```

### 1.4   Gesichtsverifikation mit Schwellenwert

Wir erinnern uns, das jedes Gesicht nun als Vektor vorliegt, wobei die einzelnen Koordinaten die n-vielen Gesichtsmerkmale
im n-dimensionalen Merkmalsraum darstellen. Diese kann die KI nun verwenden um Gesichter miteinander zu vergleichen.
Dazu benötigen wir eine Vegleichsmetrik, auch als Distanz bezeichnet. Wir verwenden hier die Kosinus Ähnlichkeit,
also der Winkel zwischen zwei Vektoren. Diese berechnet sich durch $\text{cosineSimilarity}(A, B) = 1 - \cos(A, B)$, was ihr [hier](https://towardsdatascience.com/importance-of-distance-metrics-in-machine-learning-modelling-e51395ffe60d) nochmal genauer nachlesen könnt.
 Das bedeutet die
Distanz kann Werte zwischen 0 und 1 annehmen und je näher der Wert an 0 liegt, desto ähnlicher sind sich zwei Gesichter.
An dieser Stelle kommt der Schwellenwert (_threshold_) ins Spiel. Ob die die Gesichtsverifikation zweier Gesichter als wahr
oder falsch ausgewertet wird, ist abhängig von dem Wert den wir hier festlegen. Standardmäßig lag dieser im vorangegangenen
Codeblock bei 0.4.
```python
# Festlegen des Schwellenwerts für die Gesichtsverifikation
threshold = 0.6  # Ändern Sie diesen Schwellenwert nach Bedarf

# Überprüfen Sie, ob der Wahrheitswert basierend auf dem Schwellenwert True oder False ist
updated_table_data = []
for row in table_data:
    distance = row[2]
    is_verified = distance < threshold
    updated_table_data.append([row[0], row[1], distance, is_verified])

# Aktualisierte Tabelle anzeigen
print("\nAktualisierte Tabelle mit dem Schwellenwert:")
print(tabulate(updated_table_data, headers=table_headers, tablefmt="pretty"))
```

## 2      Verwendung der live-Eingabe (hierfür benötigt ihr eine Webcam)

### 2.1   Gesichtsekennung, emotion- und age detection.

Emotion-Recognizer-Objekt: Dies ist ein Objekt, das verwendet wird, um Emotionen in Gesichtern zu erkennen.
Es ist so eingerichtet, dass es in Bildern nach Gesichtern sucht und dann die Emotionen in diesen Gesichtern 
identifiziert.

Gesichtserkennungskaskade: Dies ist ein Hilfsmittel für die Gesichtserkennung. Die "Kaskade" ist eine spezielle 
Methode, die in OpenCV (_Computer Vision Library_) verwendet wird, um Gesichter in Bildern zu finden. Diese "Kaskade" 
wurde bereits trainiert, um die Merkmale von Gesichtern zu erkennen, wie Augen, Nase und Mund. Es hilft, die 
Positionen von Gesichtern in Bildern zu finden. Der Unterschied zum Deep Learning Anzatz, der auf tiefen neuronalen 
Netzen beruht, verwendet Haar-cascade statistische Algorithmen auf handgefertigten Merkmalen. Ein Vorteil liegt darin,
dass diese Algorithmen schneller berechnet werden können und daher oft in Live-Anwendungen eingesetzt werden. Die 
geringere Genauigkeit im Vergleich zu Deep Learning Modellen nehmen wir für universellere Anwendbarkeit in Kauf.
```python
# Emotion-Recognizer-Objekt erstellen
emotion_recognizer = FER()

# Gesichtserkennungskaskade laden
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Flag für die Anzeige der Emotionen und Altersschätzung
show_emotions = True
show_age = True 
```

### 2.2 Kameraverwendung

Dieser Code erstellt also einen Echtzeit-Video-Stream, auf dem Gesichter erkannt und optional Emotionen und Altersschätzungen angezeigt werden.
    
Zuerst wird eine Endlosschleife erstellt, die kontinuierlich Bilder vom Kamerastream liest. Für jedes gelesene Bild wird 
die Gesichtserkennung durchgeführt, um Gesichter im Bild zu identifizieren. Wenn Gesichter erkannt werden:

__Wenn__ die Variable show_emotions aktiviert ist, werden Emotionen in den erkannten Gesichtern identifiziert und auf dem Bild angezeigt.
       
__Wenn__ die Variable show_age aktiviert ist, wird das Alter der erkannten Gesichter geschätzt und auf dem Bild angezeigt.
    
__Andernfalls__ wird nur ein grüner Rahmen um die erkannten Gesichter gezeichnet.


### 2.4 Shortcuts

Um eine komfortablere Anwendung zu ermöglichen, haben wir Tastaturkurzbefehle integriert, damit wir wählen können 
welche Ergebnisse angezeigt werden und die Anwendung geschlossen werden kann.

```python
# Kamera initialisieren
camera = cv2.VideoCapture(0)

while True:
    # Bild vom Kamerastream lesen
    ret, frame = camera.read()

    # Gesichtserkennung im Bild durchführen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        if show_emotions:
            # Gesichtsausschnitt extrahieren
            face_image = frame[y:y + h, x:x + w]

            # Emotionen im Gesicht erkennen
            result = emotion_recognizer.detect_emotions(face_image)

            if len(result) > 0:
                # Emotionen auf dem Bild anzeigen
                emotions = result[0]['emotions']
                dominant_emotion = max(emotions, key=emotions.get)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if show_age:
            # Alter schätzen
            age = int(w / 10)  # Einfache Schätzung basierend auf Gesichtsbreite
            cv2.putText(frame, f"AGE: {age}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        else:
            # Nur den grünen Kasten anzeigen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Emotion and Age Detection', frame)

        # Tastenabfrage
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):  # 'q' drücken, um die Schleife zu beenden
              break
        elif key == ord('e') or key == ord('E'):  # 'e' drücken, um die Emotionen ein- oder auszublenden
              show_emotions = not show_emotions
        elif key == ord('a') or key == ord('A'):  # 'a' drücken, um das Alter ein- oder auszublenden
              show_age = not show_age
 ```


### 2.5 Kameraressourcen freigeben

    Um eurer Maschine mitzuteilen, dass die Berechnung beendet ist, ist es immer wichtig, dass wir die gestarteten Prozesse sauber beenden.
```python
camera.release()
cv2.destroyAllWindows()
```

Wir hoffen, dass ihr __viel__ Freude mit dieser Anleitung habt und Interesse an den Themen Deep Learning und Gesichtserkennung gefunden habt.

PS: Falls ihr noch mehr ausprobieren und wissen wollt, schaut [hier](https://www.kaggle.com/search?q=face+detection) mal rein.

