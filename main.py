# required: pip install deepface matplotlib
import os

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from fer import FER

# Pfad zum Ordner mit den Bildern
folder_path = 'IMG_1001.JPG'

# Emotion-Recognizer-Objekt erstellen
emotion_recognizer = FER()

# Gesichtserkennungskaskade laden
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Flag für die Anzeige der Emotionen und Altersschätzung
show_emotions = True
show_age = True

# Liste, um die Ergebnisse der Gesichtsverifikation zu speichern
results = []


# Durchlaufe alle Bilder im Ordner
files = os.listdir(folder_path)
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        # Pfade zu den Bildern erstellen
        img1_path = os.path.join(folder_path, files[i])
        img2_path = os.path.join(folder_path, files[j])

        # Verifikationsprozess durchführen (mit Gesichtserkennung deaktiviert und VGG-Face-Modell)
        result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name='VGG-Face', enforce_detection=False)

        # Ergebnis zur Ergebnisliste hinzufügen
        results.append((files[i], files[j], result["distance"], result["verified"]))
        
 # Ergebnisse ausgeben
print("Bild 1\t\tBild 2\t\tDistanz\t\tWahrheitswert")
print("------------------------------------------------------------")
for filename1, filename2, distance, verified in results:
    print(f"{filename1}\t\t{filename2}\t\t{distance}\t\t{verified}")
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

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
            
            
 # Livestream anzeigen
    cv2.imshow('Emotion and Age Detection', frame)
    
  # Tastenabfrage
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):  # 'q' drücken, um die Schleife zu beenden
        break
    elif key == ord('e') or key == ord('E'):  # 'e' drücken, um die Emotionen ein- oder auszublenden
        show_emotions = not show_emotions
    elif key == ord('a') or key == ord('A'):  # 'a' drücken, um das Alter ein- oder auszublenden
        show_age = not show_age
        
 # Kameraressourcen freigeben
camera.release()
cv2.destroyAllWindows()