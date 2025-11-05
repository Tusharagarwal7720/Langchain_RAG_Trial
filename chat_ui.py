import flet as ft
import json
import os
from documentloader import load_documents, split_documents, create_vectorstore, delete_vectorstore



UPLOAD_DIR = "temp_uploads"  
HISTORY_FILE = "history.json"



def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)


def rebuild_vector_db():
    
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        return False, "No files found in upload directory."

    files = [
        os.path.join(UPLOAD_DIR, f)
        for f in os.listdir(UPLOAD_DIR)
        if f.lower().endswith((".pdf", ".txt", ".docx"))
    ]
    if not files:
        return False, "No valid files found in upload folder."

    try:
        delete_vectorstore()
        docs = load_documents(files)
        chunks = split_documents(docs)
        create_vectorstore(chunks)
        return True, f"Vector DB successfully rebuilt from {len(files)} files."
    except Exception as ex:
        return False, f"Error rebuilding vectorstore: {ex}"



def main(page: ft.Page):
    page.title = "Smart Employee Training Assistant"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.Colors.BLACK
    page.padding = 20

    success, message = rebuild_vector_db()
    print(message)


    from ragpipeline import get_qa_chain
    qa_chain = get_qa_chain()

    chat_history = []
    full_history = load_history()


    page.dialog = ft.AlertDialog(
        title=ft.Text("Knowledge Base Initialization"),
        content=ft.Text(message),
    )
    page.dialog.open = True
    page.update()

    def handle_send(e):
        question = user_input.value.strip()
        if not question:
            return

        chat_container.controls.append(
            ft.Card(
                content=ft.Container(
                    content=ft.Text(f"You: {question}", color=ft.Colors.WHITE),
                    padding=10,
                ),
                color=ft.Colors.BLUE_GREY_900,
                elevation=2,
                shape=ft.RoundedRectangleBorder(radius=10),
            )
        )
        page.update()

        try:
            response = qa_chain.invoke({"query": question})["result"]
            chat_container.controls.append(
                ft.Card(
                    content=ft.Container(
                        content=ft.Text(f"Assistant: {response}", color=ft.Colors.WHITE),
                        padding=10,
                    ),
                    color=ft.Colors.GREY_900,
                    elevation=2,
                    shape=ft.RoundedRectangleBorder(radius=10),
                )
            )
            chat_history.append({"question": question, "response": response})
            full_history.append({"question": question, "response": response})
            save_history(full_history)
        except Exception as ex:
            page.open(ft.SnackBar(ft.Text(f"Error: {str(ex)}"), bgcolor=ft.Colors.RED_700))

        user_input.value = ""
        page.update()

    def handle_clear_chat(e):
        chat_container.controls.clear()
        chat_history.clear()
        page.update()

    def handle_delete_history(e):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        full_history.clear()
        page.dialog = ft.AlertDialog(
            title=ft.Text("History Deleted"),
            content=ft.Text("All chat history has been cleared."),
        )
        page.dialog.open = True
        page.update()

    def handle_rebuild(e):
        success, msg = rebuild_vector_db()
        page.open(ft.AlertDialog(
            title=ft.Text("Rebuild Triggered"),
            content=ft.Text(msg),
        ))
        page.update()

    user_input = ft.TextField(
        hint_text="Ask a question...",
        expand=True,
        bgcolor=ft.Colors.GREY_800,
        color=ft.Colors.WHITE,
        border_radius=10,
    )

    chat_container = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10, expand=True)

    history_container = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10, expand=True)
    for item in full_history:
        history_container.controls.append(
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text(f"Q: {item['question']}", color=ft.Colors.WHITE),
                        ft.Text(f"A: {item['response']}", color=ft.Colors.WHITE),
                    ]),
                    padding=10,
                ),
                color=ft.Colors.BLUE_GREY_900,
                elevation=2,
                shape=ft.RoundedRectangleBorder(radius=10),
            )
        )


    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Home",
                icon=ft.Icons.HOME,
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "Welcome to Smart Employee Training Assistant",
                                color=ft.Colors.WHITE,
                                size=24,
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Text(
                                "Your training materials have been automatically processed.",
                                color=ft.Colors.GREY_400,
                                size=16,
                            ),
                            ft.ElevatedButton(
                                "Rebuild Knowledge Base",
                                icon=ft.Icons.REFRESH,
                                bgcolor=ft.Colors.BLUE_700,
                                color=ft.Colors.WHITE,
                                on_click=handle_rebuild,
                            ),
                            ft.ElevatedButton(
                                "Delete All History",
                                icon=ft.Icons.DELETE,
                                bgcolor=ft.Colors.RED_700,
                                color=ft.Colors.WHITE,
                                on_click=handle_delete_history,
                            ),
                            ft.ElevatedButton(
                                "Clear Current Chat",
                                icon=ft.Icons.CLEAR_ALL,
                                bgcolor=ft.Colors.GREY_700,
                                color=ft.Colors.WHITE,
                                on_click=handle_clear_chat,
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
                icon=ft.Icons.CHAT,
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                user_input,
                                ft.IconButton(ft.Icons.SEND, on_click=handle_send, tooltip="Send"),
                            ]
                        ),
                        chat_container,
                    ],
                    expand=True,
                ),
            ),
            ft.Tab(
                text="History",
                icon=ft.Icons.HISTORY,
                content=history_container,
            ),
        ],
        expand=1,
    )

    page.add(tabs)
