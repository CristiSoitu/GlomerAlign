import napari
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QFileDialog, QInputDialog
from tifffile import imread, imwrite
import numpy as np

class ImageLoader(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.loaded_layer_name = None  # Track the loaded image layer name
        
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

        # Rotate button
        self.rotate_button = QPushButton("Rotate Slice 180°")
        self.rotate_button.clicked.connect(self.rotate_slice)
        layout.addWidget(self.rotate_button)

        self.setLayout(layout)
        
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
        
        # Get the specific layer by its name
        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            print(f"Layer '{self.loaded_layer_name}' not found.")
            return

        # Prompt user for a new name and save the layer
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image As", filter="TIFF Files (*.tif *.tiff)")
        if save_path:
            imwrite(save_path, layer.data)  # Save the image data
            print(f"Image saved to {save_path}")

    def rotate_slice(self):
        if self.loaded_layer_name is None:
            print("No image has been loaded to rotate.")
            return
        
        # Get the specific layer by its name
        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            print(f"Layer '{self.loaded_layer_name}' not found.")
            return
        
        # Ensure the layer is a 3D image
        if len(layer.data.shape) != 3:
            print("Loaded layer is not a 3D image.")
            return

        # Prompt the user to select a slice to rotate
        slice_index, ok = QInputDialog.getInt(self, "Rotate Slice", "Enter slice index:", min=0, max=layer.data.shape[0] - 1)
        if not ok:
            return

        # Rotate the specified slice by 180 degrees
        layer.data[slice_index] = np.rot90(layer.data[slice_index], 2)  # Rotate 180° (2 * 90°)
        layer.refresh()  # Refresh the layer to update the viewer
        print(f"Rotated slice {slice_index} by 180°.")

def main():
    # Create two separate Napari viewers
    in_vivo_viewer = napari.Viewer(title='In Vivo Brain Viewer')
    ex_vivo_viewer = napari.Viewer(title='Ex Vivo Slices Viewer')

    # Create image loader widgets for each viewer
    in_vivo_loader = ImageLoader(in_vivo_viewer)
    ex_vivo_loader = ImageLoader(ex_vivo_viewer)
    
    # Add the loaders as dock widgets to their respective viewers
    in_vivo_viewer.window.add_dock_widget(in_vivo_loader, name='Image Loader', area='right')
    ex_vivo_viewer.window.add_dock_widget(ex_vivo_loader, name='Image Loader', area='right')

    # Run the Napari application
    napari.run()  

if __name__ == "__main__":
    main()
