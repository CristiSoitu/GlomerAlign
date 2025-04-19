import os
import numpy as np
import pandas as pd
import yaml
from tifffile import imread, imwrite
from scipy.ndimage import rotate
from skimage.measure import regionprops_table
#
import napari
from qtpy.QtWidgets import (
    QPushButton, QVBoxLayout, QWidget, QFileDialog, QInputDialog, QDialog, 
    QScrollArea, QCheckBox, QDialogButtonBox, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from cellpose import models

# Create matches directory
MATCHES_DIR = "matches"
os.makedirs(MATCHES_DIR, exist_ok=True)

# Global config variable
CONFIG = {}

def load_global_config(config_path="./config/config.yaml"):
    """Load configuration file into global CONFIG variable"""
    global CONFIG
    try:
        with open(config_path, 'r') as file:
            CONFIG = yaml.safe_load(file)
        print(f"Config loaded from {config_path}")
        print(f"Config data: {CONFIG}")
        return True
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        CONFIG = {"models": {}}
        return False

# Thread for segmentation
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
            segmented = np.array([model.eval(slice_data, channels=[2, 0], 
                                  flow_threshold=0, cellprob_threshold=0)[0] 
                                  for slice_data in self.data])
        self.finished.emit(segmented)

# Dialog for selecting slices
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
    def __init__(self, viewer, viewer_type):
        super().__init__()
        self.viewer = viewer
        self.viewer_type = viewer_type  # Either 'invivo' or 'exvivo'
        self.loaded_layer_name = None  # Track the loaded image layer name
        self.selected_slices = set()
        
        # Layout
        layout = QVBoxLayout()
        
        # Buttons for loading images and masks
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)
        
        self.load_mask_button = QPushButton("Load Mask")
        self.load_mask_button.clicked.connect(self.load_mask)
        layout.addWidget(self.load_mask_button)

        self.load_matches_button = QPushButton("Load Matched Data")
        layout.addWidget(self.load_matches_button)

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
        
        # Load config data appropriate for this viewer type
        self.load_config_data()
        
    def load_config_data(self):
        """Load data from config specific to this viewer type (invivo/exvivo)"""
        global CONFIG
        
        if not CONFIG:
            print("No configuration loaded")
            return
            
        models = CONFIG.get('models', {})
        
        # Determine which data to load based on viewer type
        if self.viewer_type == 'exvivo':
            # Load in vivo images
            if os.path.exists(models['exvivo_slices']):
                image_data = imread(models['exvivo_slices'])
                self.loaded_layer_name = 'Loaded Image'
                self.viewer.add_image(image_data, name=self.loaded_layer_name, opacity=1)
                print(f"Loaded in vivo stack from {models['exvivo_slices']}")
                
            # Load in vivo segmentation if available
            if os.path.exists(models['exvivo_segmentation']):
                mask_data = imread(models['exvivo_segmentation'])
                mask_layer = self.viewer.add_labels(mask_data, name='Mask', opacity=0.3)
                print(f"Loaded in vivo segmentation from {models['exvivo_segmentation']}")
                
        elif self.viewer_type == 'invivo':
            # Load ex vivo slices
            if os.path.exists(models['invivo_slices']):
                slices = imread(models['invivo_slices'])
                self.loaded_layer_name = 'Loaded Image'
                self.viewer.add_image(slices, name=self.loaded_layer_name, opacity=1)
                print(f"Loaded ex vivo slices from {models['invivo_slices']}")
                
            # Load ex vivo segmentation if available
            if os.path.exists(models['invivo_segmentation']):
                mask_data = imread(models['invivo_segmentation'])
                mask_layer = self.viewer.add_labels(mask_data, name='Mask', opacity=0.3)
                print(f"Loaded ex vivo segmentation from {models['invivo_segmentation']}")

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
            mask_layer = self.viewer.add_labels(mask_data, name='Mask', opacity=0.3)

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
        
        global CONFIG
        models = CONFIG.get('models', {})
        
        if not models.get('2d'):
            QMessageBox.warning(self, "Warning", "2D model path not specified in config.")
            return

        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            QMessageBox.critical(self, "Error", "Layer not found.")
            return

        self.run_segmentation(layer.data, models['2d'], is_3d=False)

    def segment_3d(self):
        if self.loaded_layer_name is None:
            QMessageBox.warning(self, "Warning", "No image loaded for segmentation.")
            return

        global CONFIG
        models = CONFIG.get('models', {})
        
        if not models.get('3d'):
            QMessageBox.warning(self, "Warning", "3D model path not specified in config.")
            return

        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            QMessageBox.critical(self, "Error", "Layer not found.")
            return

        self.run_segmentation(layer.data, models['3d'], is_3d=True)

    def run_segmentation(self, data, model_path, is_3d):
        self.worker = SegmentationWorker(data, model_path, is_3d)
        self.worker.finished.connect(self.display_segmentation_result)
        self.worker.start()

    def display_segmentation_result(self, result):
        self.viewer.add_labels(result, name="Segmentation Result")
        QMessageBox.information(self, "Segmentation Complete", "Segmentation completed and added to viewer.")


class MatchHandler:
    def __init__(self, in_vivo_viewer, ex_vivo_viewer):
        self.in_vivo_viewer = in_vivo_viewer
        self.ex_vivo_viewer = ex_vivo_viewer
        self.clicked = {'in_vivo': None, 'ex_vivo': None}
        self.glomeruli_path = os.path.join(MATCHES_DIR, 'glomeruli.csv')
        self.undo_stack = []
        self.setup()

    def setup(self):
        # Bind keys for matching and undo
        self.in_vivo_viewer.bind_key('h', self.on_key_press)
        self.ex_vivo_viewer.bind_key('h', self.on_key_press)
        self.in_vivo_viewer.bind_key('z', self.undo_match)
        self.ex_vivo_viewer.bind_key('z', self.undo_match)

    def on_key_press(self, viewer):
        # Determine which viewer was used
        viewer_name = 'in_vivo' if viewer == self.in_vivo_viewer else 'ex_vivo'
        
        # Get active layer and selected label
        active_layer = viewer.layers.selection.active
        if active_layer is None or active_layer.name != 'Mask':
            print("Please select a label from the Mask layer")
            return
            
        # Get the label at the current cursor position
        cursor_pos = tuple(map(int, np.round(viewer.cursor.position)))
        try:
            selected_label = active_layer.data[cursor_pos]
            if selected_label == 0:  # Background
                print("Background selected (label 0), please select a valid label")
                return
                
            self.on_label_selected(viewer_name, selected_label)
        except IndexError:
            print("Cursor position outside image bounds")
            
    def on_label_selected(self, viewer_name, label):
        """Store the selected label and its viewer"""
        self.clicked[viewer_name] = label
        print(f"Selected label {label} from {viewer_name} viewer")
        
        # Visual feedback
        viewer = self.in_vivo_viewer if viewer_name == 'in_vivo' else self.ex_vivo_viewer
        mask_layer = viewer.layers['Mask']
        
        # Highlight the label temporarily
        temp_data = np.zeros_like(mask_layer.data)
        temp_data[mask_layer.data == label] = 1
        viewer.add_labels(temp_data, name="Selected", opacity=0.7)
        #QMessageBox.information(viewer.window._qt_window, "Label Selected", 
        #                       f"Selected label {label}. Press 'm' in the other viewer to complete #match.")
        
        # Remove the highlight after a moment
        viewer.layers.remove('Selected')
        
        # Check if we can make a match
        other = 'ex_vivo' if viewer_name == 'in_vivo' else 'in_vivo'
        if self.clicked[other] is not None:
            self.record_match()

    def record_match(self):
        """Record a match between selected labels"""
        v1, v2 = self.clicked['in_vivo'], self.clicked['ex_vivo']
        if v1 is None or v2 is None:
            print("Need two labels selected.")
            return

        # Get the appropriate layers
        if 'Mask' not in self.in_vivo_viewer.layers or 'Mask' not in self.ex_vivo_viewer.layers:
            print("Mask layers not found in both viewers")
            return
            
        if 'matches' not in self.in_vivo_viewer.layers or 'matches' not in self.ex_vivo_viewer.layers:
            print("Matches layers not found. Please load matched data first.")
            return

        invivo_seg = self.in_vivo_viewer.layers['Mask'].data
        exvivo_seg = self.ex_vivo_viewer.layers['Mask'].data
        invivo_match = self.in_vivo_viewer.layers['matches'].data
        exvivo_match = self.ex_vivo_viewer.layers['matches'].data

        # Use invivo label as color for both matches
        color = v1

        # Update the matches layers
        invivo_match[invivo_seg == v1] = color
        exvivo_match[exvivo_seg == v2] = color

        # Refresh layers
        self.in_vivo_viewer.layers['matches'].refresh()
        self.ex_vivo_viewer.layers['matches'].refresh()

        # Visual feedback for successful match
        #QMessageBox.information(None, "Match Recorded", 
        #                       f"Matched invivo label {v1} with exvivo label {v2}")

        # Update CSV file
        if os.path.exists(self.glomeruli_path):
            df = pd.read_csv(self.glomeruli_path)
        else:
            df = pd.DataFrame(columns=['invivo', 'exvivo', 'color'])
            
        # Add new match to dataframe
        df.loc[len(df)] = [v1, v2, color]
        df.to_csv(self.glomeruli_path, index=False)

        # Save match to undo stack
        self.undo_stack.append((v1, v2, color))
        
        # Reset clicked labels
        self.clicked = {'in_vivo': None, 'ex_vivo': None}
        
        # Save updated match images
        invivo_matches_path = os.path.join(MATCHES_DIR, 'invivo_matches.tif')
        exvivo_matches_path = os.path.join(MATCHES_DIR, 'exvivo_matches.tif')
        imwrite(invivo_matches_path, invivo_match)
        imwrite(exvivo_matches_path, exvivo_match)

    def undo_match(self, viewer):
        """Undo the last match"""
        if not self.undo_stack:
            QMessageBox.information(None, "Nothing to Undo", "No matches to undo.")
            return
            
        v1, v2, color = self.undo_stack.pop()

        # Get match layers
        if 'matches' not in self.in_vivo_viewer.layers or 'matches' not in self.ex_vivo_viewer.layers:
            print("Match layers not found")
            return
            
        invivo_match = self.in_vivo_viewer.layers['matches'].data
        exvivo_match = self.ex_vivo_viewer.layers['matches'].data

        # Remove the match by setting pixels with the color back to 0
        invivo_match[invivo_match == color] = 0
        exvivo_match[exvivo_match == color] = 0
        
        # Refresh layers
        self.in_vivo_viewer.layers['matches'].refresh()
        self.ex_vivo_viewer.layers['matches'].refresh()

        # Update CSV file
        if os.path.exists(self.glomeruli_path):
            df = pd.read_csv(self.glomeruli_path)
            # Remove the match
            df = df[~((df['invivo'] == v1) & (df['exvivo'] == v2) & (df['color'] == color))]
            df.to_csv(self.glomeruli_path, index=False)
            
        # Save updated match images
        invivo_matches_path = os.path.join(MATCHES_DIR, 'invivo_matches.tif')
        exvivo_matches_path = os.path.join(MATCHES_DIR, 'exvivo_matches.tif')
        imwrite(invivo_matches_path, invivo_match)
        imwrite(exvivo_matches_path, exvivo_match)

        QMessageBox.information(None, "Match Undone", 
                               f"Undid match between invivo {v1} and exvivo {v2}")


class MatchLoader:
    def __init__(self, in_vivo_viewer, ex_vivo_viewer):
        self.in_vivo_viewer = in_vivo_viewer
        self.ex_vivo_viewer = ex_vivo_viewer

    def load_matches(self):
        """Load existing matches or create initial match files"""
        os.makedirs(MATCHES_DIR, exist_ok=True)
        base_path = os.path.join(MATCHES_DIR, 'glomeruli.csv')
        
        # Check for required mask layers
        if 'Mask' not in self.in_vivo_viewer.layers or 'Mask' not in self.ex_vivo_viewer.layers:
            QMessageBox.warning(None, "Masks Required", 
                              "Please load mask layers in both viewers first")
            return

        invivo_seg = self.in_vivo_viewer.layers['Mask'].data
        exvivo_seg = self.ex_vivo_viewer.layers['Mask'].data

        # Paths for match files
        invivo_matches_path = os.path.join(MATCHES_DIR, 'invivo_matches.tif')
        exvivo_matches_path = os.path.join(MATCHES_DIR, 'exvivo_matches.tif')
        invivo_glomeruli_path = os.path.join(MATCHES_DIR, 'invivo_glomeruli.csv')
        exvivo_glomeruli_path = os.path.join(MATCHES_DIR, 'exvivo_glomeruli.csv')

        if os.path.exists(base_path):
            print("Loading existing match data...")
            try:
                # Load match data
                invivo_data = imread(invivo_matches_path)
                exvivo_data = imread(exvivo_matches_path)
                QMessageBox.information(None, "Match Data Loaded", 
                                      "Loaded existing match data successfully")
            except Exception as e:
                print(f"Error loading match data: {e}")
                QMessageBox.warning(None, "Error", f"Error loading match data: {e}")
                return
        else:
            print("Creating initial match files...")
            try:
                # Create region tables
                invivo_df = self._get_region_table(invivo_seg)
                exvivo_df = self._get_region_table(exvivo_seg)
                
                # Save to CSV
                invivo_df.to_csv(invivo_glomeruli_path, index=False)
                exvivo_df.to_csv(exvivo_glomeruli_path, index=False)
                
                # Create empty matches CSV
                pd.DataFrame(columns=['invivo', 'exvivo', 'color']).to_csv(base_path, index=False)
                
                # Create empty match layers
                invivo_data = np.zeros_like(invivo_seg)
                exvivo_data = np.zeros_like(exvivo_seg)
                
                # Save match TIFFs
                imwrite(invivo_matches_path, invivo_data)
                imwrite(exvivo_matches_path, exvivo_data)
                
                QMessageBox.information(None, "Match Data Created", 
                                      "Created new match data files successfully")
            except Exception as e:
                print(f"Error creating match data: {e}")
                QMessageBox.warning(None, "Error", f"Error creating match data: {e}")
                return

        # Add or update match layers in viewers
        if 'matches' in self.in_vivo_viewer.layers:
            self.in_vivo_viewer.layers['matches'].data = invivo_data
            self.in_vivo_viewer.layers['matches'].refresh()
        else:
            self.in_vivo_viewer.add_labels(invivo_data, name='matches', opacity=1.0)
            
        if 'matches' in self.ex_vivo_viewer.layers:
            self.ex_vivo_viewer.layers['matches'].data = exvivo_data
            self.ex_vivo_viewer.layers['matches'].refresh()
        else:
            self.ex_vivo_viewer.add_labels(exvivo_data, name='matches', opacity=1.0)

    def _get_region_table(self, seg):
        """Extract region properties from segmentation"""
        # Handle 2D vs 3D data
        if seg.ndim == 2:
            props = regionprops_table(seg, properties=('label', 'centroid'))
            df = pd.DataFrame(props)
            df.columns = ['id', 'y', 'x']
            # Add z column with zeros for 2D data
            df['z'] = 0
        else:
            props = regionprops_table(seg, properties=('label', 'centroid'))
            df = pd.DataFrame(props)
            df.columns = ['id', 'z', 'y', 'x']
            
        # Add additional columns
        df['color'] = df['id']  # Use label ID as initial color
        df['matched'] = False   # Initial match status
        df['receptor'] = None   # Receptor type (to be filled later)
        
        return df


def main():
    """Main function to start the GlomerAlign application"""
    # Load configuration
    config_path = "./config/config.yaml"
    load_global_config(config_path)
    
    # Create the viewers
    in_vivo_viewer = napari.Viewer(title='In Vivo Brain Viewer')
    ex_vivo_viewer = napari.Viewer(title='Ex Vivo Slices Viewer')

    # Create image loaders with viewer type specification
    in_vivo_loader = ImageLoader(in_vivo_viewer, 'invivo')
    ex_vivo_loader = ImageLoader(ex_vivo_viewer, 'exvivo')
    
    # Add dock widgets
    in_vivo_viewer.window.add_dock_widget(in_vivo_loader, name='Image Loader', area='right')
    ex_vivo_viewer.window.add_dock_widget(ex_vivo_loader, name='Image Loader', area='right')

    # Create match loader and connect to buttons
    match_loader = MatchLoader(in_vivo_viewer, ex_vivo_viewer)
    in_vivo_loader.load_matches_button.clicked.connect(match_loader.load_matches)
    ex_vivo_loader.load_matches_button.clicked.connect(match_loader.load_matches)

    # Create match handler for interactions
    match_handler = MatchHandler(in_vivo_viewer, ex_vivo_viewer)

    # Run the application
    napari.run()


if __name__ == "__main__":
    main()