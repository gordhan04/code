from docx import Document
import os
import tkinter as tk
from tkinter import messagebox

def replace_partial_labels(doc_path, output_path, original_text, new_labels):
    doc = Document(doc_path)
    label_index = 0
    for para in doc.paragraphs:
        for run in para.runs:
            if original_text in run.text:
                if label_index < len(new_labels):
                    run.text = run.text.replace(original_text, new_labels[label_index])
                    label_index += 1
                else:
                    run.text = run.text.replace(original_text, "")
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        if original_text in run.text:
                            if label_index < len(new_labels):
                                run.text = run.text.replace(original_text, new_labels[label_index])
                                label_index += 1
                            else:
                                run.text = run.text.replace(original_text, "")
    doc.save(output_path)
    messagebox.showinfo("Done", f"✅ Created: {output_path} with {label_index} replaced labels.")
    os.startfile(output_path)

template_path = r"D:\Code\LABEL CHANGER\STICKER.docx"
output_path = r"D:\Code\LABEL CHANGER\newsticker.docx"
original_text = "LABEL"

# --- GUI ---
new_labels = []

def add_label():
    label = label_entry.get().strip()
    qty = qty_entry.get().strip()
    if not label:
        messagebox.showwarning("Input Error", "Label cannot be blank.")
        return
    try:
        qty = int(qty)
        if qty < 1 or qty > 48:
            messagebox.showwarning("Input Error", "Quantity must be between 1 and 48.")
            return
        new_labels.extend([label] * qty)
        label_entry.delete(0, tk.END)
        qty_entry.delete(0, tk.END)
        total_label.config(text=f"Total labels to insert: {len(new_labels)}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid number for quantity.")

def run_replace():
    if not new_labels:
        messagebox.showwarning("No Labels", "Please add at least one label.")
        return
    replace_partial_labels(template_path, output_path, original_text, new_labels)

def reset_app():
    global new_labels
    new_labels.clear()
    label_entry.delete(0, tk.END)
    qty_entry.delete(0, tk.END)
    total_label.config(text="Total labels to insert: 0")

root = tk.Tk()
root.title("Label Changer")

heading = tk.Label(root, text="Developed by Govardhan Raj", font=("Arial", 25, "bold"))
heading.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Label(frame, text="Label:").grid(row=0, column=0, padx=5)
label_entry = tk.Entry(frame)
label_entry.grid(row=0, column=1, padx=5)
label_entry.bind("<Return>", lambda event: add_label())  # Bind Enter key

tk.Label(frame, text="Quantity:").grid(row=1, column=0, padx=5)
qty_entry = tk.Entry(frame)
qty_entry.grid(row=1, column=1, padx=5)
qty_entry.bind("<Return>", lambda event: add_label())    # Bind Enter key

add_btn = tk.Button(frame, text="Add Label", command=add_label)
add_btn.grid(row=2, column=1, pady=5)

reset_btn = tk.Button(frame, text="Reset", command=reset_app, bg="red", fg="white")
reset_btn.grid(row=3, column=2, pady=5)

total_label = tk.Label(root, text="Total labels to insert: 0", font=("Arial", 12))
total_label.pack(pady=5)

run_btn = tk.Button(root, text="Replace & Open Labels", command=run_replace, bg="green", fg="white")
run_btn.pack(pady=10)

root.mainloop()