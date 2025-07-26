from django.urls import path
from . import views

app_name = 'simulacion'
urlpatterns = [
    path('', views.hello),
    path('emprendimientos/', views.lista_emprendimientos, name='lista_emprendimientos')
]