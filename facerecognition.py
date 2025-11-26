import face_recognition

known_image = face_recognition.load_image_file("person1.jpg")
unknown_image = face_recognition.load_image_file("person2.jpg")

known_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([known_encoding], unknown_encoding)
print("✅ Match:" if results[0] else "❌ No Match")
