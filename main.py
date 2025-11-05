import uvicorn
import flet as ft
from chat_ui import main
import os

if __name__ == "__main__":

    os.environ["FLET_SECRET_KEY"] = "my_local_dev_secret"
    ft.app(target=main, view=ft.WEB_BROWSER, port=8550)