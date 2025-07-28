from django.urls import path
from . import views

app_name = 'simulacion'
urlpatterns = [
    path('', views.home, name='home'),
    path('emprendimientos/', views.lista_emprendimientos, name='lista_emprendimientos'),
    path('simulacion/', views.simulacion, name='simulacion'),
    path('predicciones/', views.predicciones, name='predicciones'),
    path('evaluacion/', views.evaluacion, name='evaluacion'),
]