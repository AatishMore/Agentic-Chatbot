
import os
from tools.rag_tool import add_pdf


class RAGDocumentLoader:
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder

    def load_all(self):
        if not os.path.exists(self.pdf_folder):
            print(f"Create a '{self.pdf_folder}/' folder and place your PDFs inside it.")
        else:
            pdfs = [f for f in os.listdir(self.pdf_folder) if f.endswith(".pdf")]
            for i, pdf in enumerate(pdfs):
                add_pdf(os.path.join(self.pdf_folder, pdf), start_id=i * 1000 + 1)
            print("Done.")


PDF_FOLDER = r"C:\Users\aatish.more\Desktop\chat_agent_project\docs"

loader = RAGDocumentLoader(PDF_FOLDER)
loader.load_all()
