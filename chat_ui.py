import flet as ft
import json
import os
from ragpipeline import get_qa_chain
from documentloader import load_documents, split_documents, create_vectorstore, delete_vectorstore

qa_chain = get_qa_chain()
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def main(page: ft.Page):
    page.title = "Smart Employee Training Assistant"
    page.theme_mode = ft.ThemeMode.DARK  
    page.bgcolor = ft.colors.BLACK  
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.padding = 20

    # Load history
    chat_history = []
    full_history = load_history()

    # Upload handler
    def handle_upload(e):
        if e.files:
            files = [f.path for f in e.files]
            docs = load_documents(files)
            chunks = split_documents(docs)
            create_vectorstore(chunks)
            page.show_snack_bar(ft.SnackBar(ft.Text("Documents uploaded successfully!"), bgcolor=ft.colors.GREEN_700))
            page.update()

    # Delete handler with confirmation
    def handle_delete(e):
        def confirm_delete(e):
            delete_vectorstore()
            page.show_snack_bar(ft.SnackBar(ft.Text("Documents deleted!"), bgcolor=ft.colors.RED_700))
            dialog.open = False
            page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("Confirm Delete"),
            content=ft.Text("Delete all documents and vectorstore?"),
            actions=[
                ft.TextButton("Yes", on_click=confirm_delete),
                ft.TextButton("No", on_click=lambda e: setattr(dialog, "open", False)),
            ],
        )
        page.dialog = dialog
        dialog.open = True
        page.update()

    # Send question
    def handle_send(e):
        question = user_input.value
        if not question:
            return
        # Show user question
        chat_container.controls.append(
            ft.Card(
                content=ft.Container(
                    content=ft.Text(f"You: {question}", color=ft.colors.WHITE),
                    padding=10,
                ),
                color=ft.colors.BLUE_GREY_900,
                elevation=2,
                shape=ft.RoundedRectangleBorder(radius=10),
            )
        )
        page.update()

        # Get response
        try:
            response = qa_chain.invoke({"query": question})['result']  
            chat_container.controls.append(
                ft.Card(
                    content=ft.Container(
                        content=ft.Text(f"Assistant: {response}", color=ft.colors.WHITE),
                        padding=10,
                    ),
                    color=ft.colors.GREY_900,
                    elevation=2,
                    shape=ft.RoundedRectangleBorder(radius=10),
                )
            )
            # Save to history
            chat_history.append({"question": question, "response": response})
            full_history.append({"question": question, "response": response})
            save_history(full_history)
        except Exception as ex:
            page.show_snack_bar(ft.SnackBar(ft.Text(f"Error: {str(ex)}"), bgcolor=ft.colors.RED_700))

        user_input.value = ""
        page.update()

    # Clear chat
    def handle_clear_chat(e):
        chat_container.controls.clear()
        chat_history.clear()
        page.update()

    # UI Components
    file_picker = ft.FilePicker(on_result=handle_upload)

    user_input = ft.TextField(
        hint_text="Ask a question...",
        expand=True,
        bgcolor=ft.colors.GREY_800,
        color=ft.colors.WHITE,
        border_radius=10,
    )

    chat_container = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10, expand=True)

    history_container = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10, expand=True)
    for item in full_history:
        history_container.controls.append(
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text(f"Q: {item['question']}", color=ft.colors.WHITE),
                        ft.Text(f"A: {item['response']}", color=ft.colors.WHITE),
                    ]),
                    padding=10,
                ),
                color=ft.colors.BLUE_GREY_900,
                elevation=2,
                shape=ft.RoundedRectangleBorder(radius=10),
            )
        )

    # Tabs for navigation
    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Home",
                icon=ft.icons.HOME,
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.ElevatedButton(
                                "Upload Documents",
                                icon=ft.icons.UPLOAD_FILE,
                                on_click=lambda _: file_picker.pick_files(allow_multiple=True),
                                bgcolor=ft.colors.BLUE_700,
                                color=ft.colors.WHITE,
                                tooltip="Upload PDF/TXT/DOCX files",
                            ),
                            ft.ElevatedButton(
                                "Delete Documents",
                                icon=ft.icons.DELETE,
                                on_click=handle_delete,
                                bgcolor=ft.colors.RED_700,
                                color=ft.colors.WHITE,
                                tooltip="Delete all uploaded documents",
                            ),
                            ft.ElevatedButton(
                                "Clear Chat",
                                icon=ft.icons.CLEAR_ALL,
                                on_click=handle_clear_chat,
                                bgcolor=ft.colors.GREY_700,
                                color=ft.colors.WHITE,
                                tooltip="Clear current chat",
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=20,
                    ),
                    alignment=ft.alignment.center,
                    padding=50,
                ),
            ),
            ft.Tab(
                text="Chat",
                icon=ft.icons.CHAT,
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                user_input,
                                ft.IconButton(ft.icons.SEND, on_click=handle_send, tooltip="Send"),
                            ]
                        ),
                        chat_container,
                    ],
                    expand=True,
                ),
            ),
            ft.Tab(
                text="History",
                icon=ft.icons.HISTORY,
                content=history_container,
            ),
        ],
        expand=1,
    )

    page.add(file_picker, tabs)
