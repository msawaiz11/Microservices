
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Main_Dish.settings")

import django
django.setup()


from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from Main_Dish_App.consumers import TaskStatusConsumer

# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Main_Dish.settings")

# application = get_asgi_application()

websocket_urlpattern = [
  path("ws/main_dish/", TaskStatusConsumer.as_asgi()),
]

application = ProtocolTypeRouter({
  "http" : get_asgi_application(),
  "websocket" : URLRouter(websocket_urlpattern)
})