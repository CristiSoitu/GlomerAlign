import napari
from qtpy.QtWidgets import (
    QPushButton, QVBoxLayout, QWidget, QFileDialog, QInputDialog, QDialog, QScrollArea, QVBoxLayout, QCheckBox, QDialogButtonBox, QHBoxLayout, QMessageBox
)
from tifffile import imread, imwrite
import numpy as np
from scipy.ndimage import rotate
from PyQt5.QtCore import QThread, pyqtSignal
import yaml
from cellpose import models


class SegmentationWorker(QThread):
    finished = pyqtSignal(np.ndarray)

    def __init__(self, data, model_path, is_3d=False):
        super().__init__()
        self.data = data
        self.model_path = model_path
        self.is_3d = is_3d

    def run(self):
        model = models.CellposeModel(gpu=True, pretrained_model=self.model_path)
        if self.is_3d:
            segmented = model.eval(self.data, channels=[0, 0], do_3D=True)[0]
        else:
            segmented = np.array([model.eval(slice, channels=[0, 0])[0] for slice in self.data])
        self.finished.emit(segmented)

class SliceSelectorDialog(QDialog):
    def __init__(self, num_slices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Slices")
        self.selected_slices = set()
        
        # Scrollable area for slices
        layout = QVBoxLayout()
        scroll_area = QScrollArea(self)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.checkboxes = []
        for i in range(num_slices):
            checkbox = QCheckBox(f"Slice {i}")
            checkbox.stateChanged.connect(self.update_selection)
            scroll_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)
        
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Select All / Deselect All buttons
        button_layout = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_button)

        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(self.deselect_all)
        button_layout.addWidget(deselect_all_button)
        layout.addLayout(button_layout)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)

    def update_selection(self):
        self.selected_slices = {
            i for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()
        }

    def select_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def deselect_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)


class ImageLoader(QWidget):
    def __init__(self, viewer, config_path="config/config.yaml"):
        super().__init__()
        self.viewer = viewer
        self.loaded_layer_name = None  # Track the loaded image layer name
        self.model_paths = self.load_config(config_path)

        
        # Layout
        layout = QVBoxLayout()
        
        # Buttons for loading images and masks
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)
        
        self.load_mask_button = QPushButton("Load Mask")
        self.load_mask_button.clicked.connect(self.load_mask)
        layout.addWidget(self.load_mask_button)

        # Save button
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        layout.addWidget(self.save_button)

        # Select slices button
        self.select_slices_button = QPushButton("Select Slices")
        self.select_slices_button.clicked.connect(self.select_slices)
        layout.addWidget(self.select_slices_button)

        # Segmentation buttons
        self.segment_2d_button = QPushButton("Segmentation 2D")
        self.segment_2d_button.clicked.connect(self.segment_2d)
        layout.addWidget(self.segment_2d_button)

        self.segment_3d_button = QPushButton("Segmentation 3D")
        self.segment_3d_button.clicked.connect(self.segment_3d)
        layout.addWidget(self.segment_3d_button)

        # Transform buttons
        self.rotate_180_button = QPushButton("Rotate 180°")
        self.rotate_180_button.clicked.connect(self.rotate_180)
        layout.addWidget(self.rotate_180_button)

        self.rotate_90_button = QPushButton("Rotate 90°")
        self.rotate_90_button.clicked.connect(self.rotate_90)
        layout.addWidget(self.rotate_90_button)

        self.flip_horizontal_button = QPushButton("Flip Horizontally")
        self.flip_horizontal_button.clicked.connect(self.flip_horizontal)
        layout.addWidget(self.flip_horizontal_button)

        self.flip_vertical_button = QPushButton("Flip Vertically")
        self.flip_vertical_button.clicked.connect(self.flip_vertical)
        layout.addWidget(self.flip_vertical_button)

        self.custom_rotate_button = QPushButton("Rotate by Custom Angle")
        self.custom_rotate_button.clicked.connect(self.rotate_custom)
        layout.addWidget(self.custom_rotate_button)

        self.setLayout(layout)

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config['models']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config file: {e}")
            return {"2d": None, "3d": None}        
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", filter="TIFF Files (*.tif *.tiff)")
        if file_path:
            image_data = imread(file_path)  # Load the TIFF image
            self.loaded_layer_name = 'Loaded Image'
            self.viewer.add_image(image_data, name=self.loaded_layer_name)

    def load_mask(self):
        mask_path, _ = QFileDialog.getOpenFileName(self, "Open Mask File", filter="TIFF Files (*.tif *.tiff)")
        if mask_path:
            mask_data = imread(mask_path)  # Load the TIFF mask
            self.viewer.add_labels(mask_data, name='Mask')

    def save_image(self):
        if self.loaded_layer_name is None:
            print("No image has been loaded to save.")
            return
        
        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            print(f"Layer '{self.loaded_layer_name}' not found.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image As", filter="TIFF Files (*.tif *.tiff)")
        if save_path:
            imwrite(save_path, layer.data)
            print(f"Image saved to {save_path}")

    def select_slices(self):
        if self.loaded_layer_name is None:
            print("No image loaded to select slices.")
            return
        
        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            print(f"Layer '{self.loaded_layer_name}' not found.")
            return

        # Open the slice selector dialog
        dialog = SliceSelectorDialog(num_slices=layer.data.shape[0])
        if dialog.exec_():
            self.selected_slices = dialog.selected_slices
            print(f"Selected slices: {self.selected_slices}")

    def rotate_180(self):
        self.apply_transformation(lambda data: np.rot90(data, 2))

    def rotate_90(self):
        self.apply_transformation(lambda data: np.rot90(data, 1))

    def flip_horizontal(self):
        self.apply_transformation(np.fliplr)

    def flip_vertical(self):
        self.apply_transformation(np.flipud)


    def rotate_custom(self):
        if not self.selected_slices:
            print("No slices selected for transformation.")
            return
        
        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            print(f"Layer '{self.loaded_layer_name}' not found.")
            return

        # Prompt the user for the custom angle
        angle, ok = QInputDialog.getDouble(self, "Rotate Slices", "Enter rotation angle (degrees):", 0, -360, 360, 1)
        if not ok:
            return

        for slice_idx in self.selected_slices:
            layer.data[slice_idx] = rotate(layer.data[slice_idx], angle, reshape=False, mode='nearest')

        layer.refresh()  # Update the viewer
        print(f"Rotated slices: {self.selected_slices} by {angle}°")

    def apply_transformation(self, transform):
        if not self.selected_slices:
            print("No slices selected for transformation.")
            return

        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            print(f"Layer '{self.loaded_layer_name}' not found.")
            return

        for slice_idx in self.selected_slices:
            layer.data[slice_idx] = transform(layer.data[slice_idx])

        layer.refresh()  # Update the viewer
        print(f"Transformed slices: {self.selected_slices}")


    def segment_2d(self):
        if self.loaded_layer_name is None:
            QMessageBox.warning(self, "Warning", "No image loaded for segmentation.")
            return
        
        if not self.model_paths['2d']:
            QMessageBox.warning(self, "Warning", "2D model path not specified in config.")
            return

        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            QMessageBox.critical(self, "Error", "Layer not found.")
            return

        self.run_segmentation(layer.data, self.model_paths['2d'], is_3d=False)

    def segment_3d(self):
        if self.loaded_layer_name is None:
            QMessageBox.warning(self, "Warning", "No image loaded for segmentation.")
            return

        if not self.model_paths['3d']:
            QMessageBox.warning(self, "Warning", "3D model path not specified in config.")
            return

        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            QMessageBox.critical(self, "Error", "Layer not found.")
            return

        self.run_segmentation(layer.data, self.model_paths['3d'], is_3d=True)

    def run_segmentation(self, data, model_path, is_3d):
        self.worker = SegmentationWorker(data, model_path, is_3d)
        self.worker.finished.connect(self.display_segmentation_result)
        self.worker.start()

    def display_segmentation_result(self, result):
        self.viewer.add_labels(result, name="Segmentation Result")
        QMessageBox.information(self, "Segmentation Complete", "Segmentation completed and added to viewer.")



def main():
    in_vivo_viewer = napari.Viewer(title='In Vivo Brain Viewer')
    ex_vivo_viewer = napari.Viewer(title='Ex Vivo Slices Viewer')

    in_vivo_loader = ImageLoader(in_vivo_viewer)
    ex_vivo_loader = ImageLoader(ex_vivo_viewer)
    
    in_vivo_viewer.window.add_dock_widget(in_vivo_loader, name='Image Loader', area='right')
    ex_vivo_viewer.window.add_dock_widget(ex_vivo_loader, name='Image Loader', area='right')

    napari.run()  


if __name__ == "__main__":
    main()
