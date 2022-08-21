import os
import cv2
import face_recognition


video_file = cv2.VideoCapture(os.path.abspath("videos/sample_video.mp4"))
cascPath = os.path.abspath("haarcascade_frontalface_default.xml")

length = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))


image_thomas_shelby_1 = face_recognition.load_image_file(
    os.path.abspath("images/thomas_shelby.jpeg"))
image_john_shelby_1 = face_recognition.load_image_file(
    os.path.abspath("images/john_shelby.jpg"))
image_arthur_shelby_1 = face_recognition.load_image_file(
    os.path.abspath("images/arthur_shelby.jpg"))
image_michael_shelby_1 = face_recognition.load_image_file(
    os.path.abspath("images/michael_shelby.jpg"))


thomas_shelby_face_1 = face_recognition.face_encodings(image_thomas_shelby_1)[
    0]
john_shelby_face_1 = face_recognition.face_encodings(image_john_shelby_1)[0]
arthur_shelby_face_1 = face_recognition.face_encodings(image_arthur_shelby_1)[
    0]
michael_shelby_face_1 = face_recognition.face_encodings(
    image_michael_shelby_1)[0]


known_faces = [
    thomas_shelby_face_1, john_shelby_face_1, arthur_shelby_face_1, michael_shelby_face_1
]

facial_points = []
face_encodings = []
facial_number = 0

while True:
    return_value, frame = video_file.read()
    facial_number = facial_number + 1

    if not return_value:
        break
    rgb_frame = frame[:, :, ::-1]

    facial_points = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, facial_points)

    facial_names = []
    for encoding in face_encodings:
        match = face_recognition.compare_faces(
            known_faces, encoding, tolerance=0.50)

        name = ""
        if match[0]:
            name = "Thomas Shelby"
        if match[1]:
            name = "John Shelby"
        if match[2]:
            name = "Arthur Shelby"
        if match[3]:
            name = "Michael Shelby"

        facial_names.append(name)

    for (top, right, bottom, left), name in zip(facial_points, facial_names):

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 25),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 0.5, (255, 255, 255), 1)

    codec = int(video_file.get(cv2.CAP_PROP_FOURCC))
    fps = int(video_file.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_movie = cv2.VideoWriter("output_{}.mp4".format(
        facial_number), codec, fps, (frame_width, frame_height))
    print("Writing frame {} / {}".format(facial_number, length))
    output_movie.write(frame)

video_file.release()
output_movie.release()
cv2.destroyAllWindows()
