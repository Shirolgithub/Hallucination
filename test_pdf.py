from models.pdf_reader import extract_text_from_pdf

pdf_path = "sample.pdf"

text = extract_text_from_pdf(pdf_path)

print("\n===== EXTRACTED PDF TEXT =====\n")

print(text[:2000])