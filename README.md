# Label Changer Project

## Overview
The Label Changer project is a Python application designed to facilitate the replacement of labels in Word documents. It allows users to specify new labels and automatically replaces occurrences of a designated original text within the document.

## Files
- **label_maker.py**: Contains the main functionality for replacing labels in a Word document. It defines the function `replace_partial_labels`, which processes both paragraphs and tables to replace specified text.
  
- **requirements.txt**: Lists the dependencies required for the project. Currently, it includes the `python-docx` library, which is essential for manipulating Word documents.

## Usage Instructions
1. Ensure you have Python installed on your machine.
2. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```
3. Open the `label_maker.py` file and modify the `template_path`, `output_path`, and `original_text` variables as needed.
4. Run the script:
   ```
   python label_maker.py
   ```
5. Follow the prompts to enter new labels and specify how many times each label should be repeated.

## Contribution
Feel free to contribute to the project by submitting issues or pull requests. Your feedback and suggestions are welcome!

## License
This project is open-source and available for modification and distribution under the terms of the MIT License.