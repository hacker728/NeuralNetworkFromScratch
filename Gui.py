import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2

class TextDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Text Detection Application")

        # Styling
        self.style = ttk.Style()
        self.style.configure("TButton", padding=5, relief="flat", background="#007bff", foreground="white")
        self.style.configure("TLabel", background="#f8f9fa", foreground="#495057")
        self.style.configure("TFrame", background="#f8f9fa")
        self.style.configure("TText", background="white", foreground="#495057")

        # Create widgets
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(pady=20)

        self.upload_button = ttk.Button(self.main_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.grid(row=0, column=0, padx=10, pady=10)

        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=1, column=0, padx=10, pady=10)

        self.text_output_label = ttk.Label(self.main_frame, text="Detected Text:")
        self.text_output_label.grid(row=2, column=0, padx=10, pady=5)

        self.text_output_text = tk.Text(self.main_frame, height=10, width=50)
        self.text_output_text.grid(row=3, column=0, padx=10, pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            try:
                image = Image.open(file_path)
                image.thumbnail((300, 300))  # Resize image for display
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo

                # Process image to detect text (replace this with your text detection code)
                detected_text = self.detect_text(image)
                self.text_output_text.delete(1.0, tk.END)  # Clear previous text
                self.text_output_text.insert(tk.END, detected_text)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")

    def detect_text(self, image):
        # Placeholder for text detection (replace this with your actual neural network code)
        text_regions = self.extract_text_regions(image)

        # Sort each row of pixels and group them into letters
        sorted_letters = []
        for region in text_regions:
            sorted_row = [sorted(row) for row in region]  # Sort each row of pixels
            sorted_letters.append(sorted_row)

        # Flatten the sorted letters into a 1D list
        flat_sorted_letters = np.concatenate(sorted_letters)

        # Feed each letter to your trained model for inference (replace this with your inference code)
        return "Detected text will appear here"

    def extract_text_regions(self, image):
        # Convert image to grayscale using OpenCV
        grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Apply thresholding to extract text regions using OpenCV
        _, thresholded_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY_INV)

        # Perform connected component analysis to identify text regions using OpenCV
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image, connectivity=8)

        # Extract text regions based on connected component analysis results
        text_regions = []
        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            text_region = image.crop((x, y, x + w, y + h))
            text_regions.append(text_region)

        return text_regions

def main():
    root = tk.Tk()
    app = TextDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
