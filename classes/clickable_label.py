from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen

class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.points_number = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap() is not None:
            x = event.pos().x()
            y = event.pos().y()

            label_width = self.width()
            label_height = self.height()

            pixmap_width = self.pixmap().width()
            pixmap_height = self.pixmap().height()

            scale_x = pixmap_width / label_width
            scale_y = pixmap_height / label_height

            img_x = int(x * scale_x)
            img_y = int(y * scale_y)

            print(f"Clicked at label ({x}, {y}), mapped to image ({img_x}, {img_y})")
            if(len(self.points) < self.points_number):
                self.points.append((img_x, img_y))
            else:
                self.points.pop(0)
                self.points.append((img_x, img_y))
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 5))
        for point in self.points:
            # Map back to label coordinates for drawing
            label_width = self.width()
            label_height = self.height()
            pixmap_width = self.pixmap().width()
            pixmap_height = self.pixmap().height()

            scale_x = label_width / pixmap_width
            scale_y = label_height / pixmap_height

            draw_x = int(point[0] * scale_x)
            draw_y = int(point[1] * scale_y)
            painter.drawPoint(draw_x, draw_y)

    def get_points(self):
        return self.points

    def clear_points(self):
        self.points = []
        self.update()