import sys
import os
import cv2
import shutil
import copy
import torch
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow, QWidget,
    QLabel, QFileDialog,
    QMessageBox, QHBoxLayout,
    QInputDialog,
    QPushButton,
    QTabWidget,
    QVBoxLayout, QAction)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
from detector import FaceDetector
from attribute import FaceAttribute
from recognizer import FaceRecognizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

style_sheel = """
    QLabel#ImageLabel{
        color: darkgrey;
        border: 2px solid #000000;
        qproperty-alignment: AlignCenter
    }
"""


class VideoWorkerThread(QThread):
    frame_data_updated = pyqtSignal(np.ndarray)
    invalid_video_file = pyqtSignal()

    def __init__(self, parent, video_file=None):
        super().__init__()
        self.parent = parent
        self.video_file = video_file

    def run(self):
        self.capture = cv2.VideoCapture(self.video_file)

        if self.video_file != 0:
            self.capture.set(cv2.CAP_PROP_FPS, 1/60)

        if not self.capture.isOpened():
            self.invalid_video_file.emit()
        else:
            while self.parent.thread_is_running:
                ret_val, frame = self.capture.read()

                if not ret_val:
                    break
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)

                    if self.parent.isdetect or self.parent.isattribute:
                        boxes = self.parent.face_detector.detect_face(frame)

                        if boxes is not None:

                            for box in boxes:
                                x1, y1, x2, y2 = box
                                if self.parent.isattribute:
                                    age, gender = self.parent.attribute_detector.detect_age_gender(
                                        image=frame, box=box)
                                    cv2.putText(frame, f"{gender},{age}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                                cv2.rectangle(
                                    frame, (x1, y1), (x2, y2), (0, 225, 0), 2)
                    elif self.parent.isrecognize:
                        details = self.parent.recognizer.recognize(frame)

                        for name, min_dist, box in details:

                            x1, y1, x2, y2 = box

                            cv2.putText(frame,
                                        f"{name} {min_dist:.3f}", (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 1)
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)
                    self.frame_data_updated.emit(frame)

    def stopThread(self):
        self.capture.release()
        self.wait()
        QApplication.processEvents()


class NeuralFace(QMainWindow):
    def __init__(self):
        super().__init__()

        self.thread_is_running = False

        self.face_detector = FaceDetector(model="mtcnn", device=device)
        self.attribute_detector = FaceAttribute()
        self.recognizer = FaceRecognizer(device=device)

        self.isdetect = False
        self.isattribute = False
        self.isrecognize = False

        self.initializeUI()

    def initializeUI(self):
        self.setMinimumSize(900, 600)
        self.setWindowTitle("Neural Face")

        self.setupWindow()
        self.setupMenu()
        self.show()

    def setupWindow(self):
        self.image_label = QLabel()
        self.image_label.setObjectName("ImageLabel")

        self.tabs = QTabWidget()
        self.image_tab = QWidget()
        self.video_tab = QWidget()

        self.tabs.addTab(self.image_tab, "Image")
        self.tabs.addTab(self.video_tab, "Video")


        self.message = QLabel()

        self.detect_face_button = QPushButton("Detect Face")
        self.detect_face_button.setEnabled(False)
        self.detect_face_button.clicked.connect(self.detection)

        self.face_attribute_button = QPushButton("Detect Face, Age, Gender")
        self.face_attribute_button.setEnabled(False)
        self.face_attribute_button.clicked.connect(self.attribute_detection)

        self.recognize_button = QPushButton("Recognize")
        self.recognize_button.setEnabled(False)
        self.recognize_button.clicked.connect(self.recognize)

        self.register_button = QPushButton("Register Face")
        self.register_button.setEnabled(True)
        self.register_button.clicked.connect(self.register_image)

        self.register_video_button = QPushButton("Register Face")
        self.register_video_button.setEnabled(True)
        self.register_video_button.clicked.connect(self.register_image)

        self.start_button = QPushButton("Start WebCam")
        self.start_button.clicked.connect(self.startVideo)

        self.detect_video_button = QPushButton("Detect Face")
        self.detect_video_button.setEnabled(False)
        self.detect_video_button.clicked.connect(self.live_detection)

        self.attribut_video_button = QPushButton("Estimate Age, Gender")
        self.attribut_video_button.setEnabled(False)
        self.attribut_video_button.clicked.connect(self.live_attribute)

        self.recognize_video_button = QPushButton("Recognize")
        self.recognize_video_button.setEnabled(False)
        self.recognize_video_button.clicked.connect(self.live_recognize)

        self.stop_video_button = QPushButton("Stop Stream")
        self.stop_video_button.setEnabled(False)
        self.stop_video_button.clicked.connect(self.stopCurrentVideo)

        side_panel_v_box = QVBoxLayout()
        side_panel_v_box.setAlignment(Qt.AlignmentFlag.AlignTop)
        side_panel_v_box.addWidget(
            QLabel("Select image through menu bar or Ctrl+O"))
        side_panel_v_box.addWidget(self.message)
        side_panel_v_box.addWidget(self.detect_face_button)
        side_panel_v_box.addSpacing(10)
        side_panel_v_box.addWidget(self.face_attribute_button)
        side_panel_v_box.addSpacing(10)
        side_panel_v_box.addWidget(self.recognize_button)
        side_panel_v_box.addStretch(1)
        side_panel_v_box.addWidget(self.register_button)

        side_panel_video_box = QVBoxLayout()
        side_panel_video_box.setAlignment(Qt.AlignmentFlag.AlignTop)
        side_panel_video_box.addWidget(QLabel("For Webcam "))
        side_panel_video_box.addWidget(self.start_button)
        side_panel_video_box.addSpacing(10)
        side_panel_video_box.addWidget(self.detect_video_button)
        side_panel_video_box.addSpacing(10)
        side_panel_video_box.addWidget(self.attribut_video_button)
        side_panel_video_box.addSpacing(10)
        side_panel_video_box.addWidget(self.recognize_video_button)
        side_panel_video_box.addSpacing(10)
        side_panel_video_box.addWidget(self.stop_video_button)
        side_panel_video_box.addStretch(1)
        side_panel_video_box.addWidget(self.register_video_button)

        self.image_tab.setMinimumWidth(200)
        self.image_tab.setLayout(side_panel_v_box)

        self.video_tab.setMinimumWidth(200)
        self.video_tab.setLayout(side_panel_video_box)

        main_h_box = QHBoxLayout()
        main_h_box.addWidget(self.image_label, 1)
        main_h_box.addWidget(self.tabs)

        container = QWidget()
        container.setLayout(main_h_box)
        self.setCentralWidget(container)

    def setupMenu(self):
        open_act = QAction('Open...', self)
        open_act.setShortcut('Ctrl+O')
        open_act.triggered.connect(self.openImageFile)

        open_video_act = QAction('Open Video', self)
        open_video_act.triggered.connect(self.openVideoFile)

        save_act = QAction('Save...', self)
        save_act.setShortcut('Ctrl+S')
        save_act.triggered.connect(self.saveImageFile)

        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        file_menu = menu_bar.addMenu("File")
        file_menu.addActions([open_act, open_video_act, save_act])

    def startVideo(self):
        self.thread_is_running = True
        self.start_button.setEnabled(False)
        self.video_thread_worker = VideoWorkerThread(self, 0)

        self.startWorker()

    def startWorker(self):
        self.video_thread_worker.frame_data_updated.connect(
            self.convertCVToQImage)
        self.video_thread_worker.invalid_video_file.connect(
            self.invalidVideoFile)
        self.video_thread_worker.start()

        self.stop_video_button.setEnabled(True)
        self.detect_video_button.setEnabled(True)
        self.attribut_video_button.setEnabled(True)
        self.recognize_video_button.setEnabled(True)

    def stopCurrentVideo(self):
        self.stop_video_button.setEnabled(False)
        self.detect_video_button.setEnabled(False)
        self.attribut_video_button.setEnabled(False)
        self.recognize_video_button.setEnabled(False)
        if self.thread_is_running:
            self.thread_is_running = False
            self.video_thread_worker.stopThread()

            self.image_label.clear()
            self.start_button.setEnabled(True)

    def live_detection(self):
        if self.thread_is_running:
            self.isdetect = not self.isdetect
            self.isattribute = False
            self.isrecognize = False

    def live_attribute(self):
        if self.thread_is_running:
            self.isdetect = False
            self.isattribute = not self.isattribute
            self.isrecognize = False

    def live_recognize(self):
        if self.thread_is_running:
            self.isdetect = False
            self.isattribute = False
            self.isrecognize = not self.isrecognize

    def detection(self):
        # self.cv_image = self.copy_cv_image
        self.cv_image = copy.deepcopy(self.copy_cv_image)
        boxes = self.face_detector.detect_face(self.cv_image)

        if boxes is not None:
            self.message.setText(f"{len(boxes)} Face Found")
            self.message.repaint()
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(self.cv_image, (x1, y1),
                              (x2, y2), (0, 255, 0), 2)
        else:
            self.message.setText("No Face Found")
            self.message.repaint()

        self.convertCVToQImage(self.cv_image)

        self.image_label.repaint()

    def attribute_detection(self):
        self.cv_image = copy.deepcopy(self.copy_cv_image)

        boxes = self.face_detector.detect_face(self.cv_image)

        if boxes is not None:
            self.message.setText(f"{len(boxes)} Face Found")
            self.message.repaint()
            for box in boxes:
                age, gender = self.attribute_detector.detect_age_gender(
                    image=self.cv_image, box=box)

                x1, y1, x2, y2 = box
                cv2.rectangle(self.cv_image, (x1, y1),
                              (x2, y2), (0, 225, 0), 2)

                cv2.putText(self.cv_image, f"{gender},{age}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            self.message.setText("No Face Found")
            self.message.repaint()

        self.convertCVToQImage(self.cv_image)
        self.image_label.repaint()

    def recognize(self):
        self.cv_image = copy.deepcopy(self.copy_cv_image)

        image = Image.fromarray(self.cv_image)

        details = self.recognizer.recognize(image)

        self.message.setText(f"{len(details)} faces found")
        self.message.repaint()

        for name, min_dist, box in details:

            x1, y1, x2, y2 = box
            text = f"{name},{min_dist:.2f}"

            cv2.putText(self.cv_image,
                        text, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(self.cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.convertCVToQImage(self.cv_image)
        self.image_label.repaint()

    def register_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", os.getenv('HOME'),
                                                    "All Files (*);; Images (*.png, *jpeg, *jpg, *.bmp)")

        if image_path:
            name, done = QInputDialog.getText(
                self, "Get Name", "Enter name under which image to be registered: ")

            if done and len(name) > 0:
                image_file = image_path.split('/')[-1]
                print(f"Registering {name}/{image_file} image")

                os.mkdir(f"saved/{name}")
                shutil.copy(image_path, f"saved/{name}/{image_file}")
                self.recognizer.register()
            else:
                QMessageBox.information(
                    self, "Alert", "No Input provided", QMessageBox.StandardButton.Ok)
        else:
            QMessageBox.information(
                self, "Alert", "No Image provided", QMessageBox.StandardButton.Ok)

    def openImageFile(self):
        self.stopCurrentVideo()

        self.tabs.setCurrentIndex(self.tabs.indexOf(self.image_tab))
        image_file, _ = QFileDialog.getOpenFileName(self, "Open Image",
                                                    os.getenv('HOME'), "All Files (*);; Images (*.png, *.jpeg, *.jpg, *.bmp)")

        if image_file:
            self.detect_face_button.setEnabled(True)
            self.face_attribute_button.setEnabled(True)
            self.recognize_button.setEnabled(True)
            image = cv2.imread(image_file)
            self.cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.copy_cv_image = copy.deepcopy(self.cv_image)

            self.convertCVToQImage(self.cv_image)
        else:
            QMessageBox.information(
                self, 'Error', "No image was loaded", QMessageBox.StandardButton.Ok)

    def openVideoFile(self):
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.video_tab))
        video_file, _ = QFileDialog.getOpenFileName(self, "Open Video",
                                                    os.getenv(
                                                        'HOME'), "All files(*);;Videos (*.mp4, *.avi)"
                                                    )

        if video_file:
            self.video_path_line = video_file
            self.thread_is_running = True
            self.start_button.setEnabled(False)

            self.video_thread_worker = VideoWorkerThread(self, video_file)

            self.startWorker()
        else:
            QMessageBox.information(
                self, 'Error', "No video was loaded.", QMessageBox.StandardButton.Ok)

        # self.start_button.repaint()

    def saveImageFile(self):
        image_file, _ = QFileDialog.getSaveFileName(self, "Save Image", os.getenv('HOME'),
                                                    "JPEG (*.jpeg);; JPG (*.jpg);; PNG (*.png);;Bitmap (*.bmp)")

        if image_file and self.image_label.pixmap() is not None:
            cv2.imwrite(image_file, self.cv_image)
        else:
            QMessageBox.information(
                self, "Error", "Unable to save image.", QMessageBox.StandardButton.Ok)

    def convertCVToQImage(self, cv_image):
        height, width, channels = cv_image.shape
        bytes_per_line = width * channels
        converted_Qt_image = QImage(
            cv_image, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        self.image_label.setPixmap(QPixmap.fromImage(converted_Qt_image)
                                   .scaled(self.image_label.width(), self.image_label.height(),
                                           Qt.AspectRatioMode.KeepAspectRatio))

    def invalidVideoFile(self):
        QMessageBox.warning(self, "Error", "No Video was loaded",
                            QMessageBox.StandardButton.Ok)
        self.start_button.setEnabled(True)
        self.video_path_line = ""

    def closeEvent(self, event):
        if self.thread_is_running:
            self.video_thread_worker.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(style_sheel)
    window = NeuralFace()
    sys.exit(app.exec_())
