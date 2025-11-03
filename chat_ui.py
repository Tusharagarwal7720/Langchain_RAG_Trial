import flet as ft
from rag_pipeline import get_qa_chain
from document_loader import load_documents, split_documents, create_vectorstore
import os

qa_chain = get_qa_chain()

def main(page: ft.Page):
    page.title = "Smart Employee Training Assistant"
    page.vertical_alignment = ft.MainAxisAlignment.START

    chat_history = ft.Column()
    user_input = ft.TextField(hint_text="Ask a question...", expand=True)
    
    def handle_upload(e):
        if e.files:
            files = [f.path for f in e.files]
            docs = load_documents(files)
            chunks = split_documents(docs)
            create_vectorstore(chunks)
            page.snack_bar = ft.SnackBar(ft.Text("Documents uploaded successfully!"))
            page.snack_bar.open = True
            page.update()

    def handle_send(e):
        question = user_input.value
        if not question:
            return
        chat_history.controls.append(ft.Row([ft.Text("You:", weight="bold"), ft.Text(question)]))
        response = qa_chain.run(question)
        chat_history.controls.append(ft.Row([ft.Text("Assistant:", weight="bold"), ft.Text(response)]))
        user_input.value = ""
        page.update()

    page.add(
        ft.FilePicker(on_result=handle_upload),
        ft.Row([user_input, ft.IconButton(ft.icons.SEND, on_click=handle_send)]),
        chat_history
    )

