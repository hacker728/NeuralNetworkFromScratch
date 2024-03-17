import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

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
        return "Detected text will appear here"

def main():
    root = tk.Tk()
    app = TextDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
