import PyPDF2

def extract_text_from_pdf(pdf_path, txt_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write(text)
    return text

if __name__ == "__main__":
    extract_text_from_pdf('sample_specs/PDF/slmp.pdf', 'sample_specs/Markdown/slmp.txt')