import uvicorn
import flet as ft
from chat_ui import main

if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port=8550)