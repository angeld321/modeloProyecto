from django.urls import path
from . import views

app_name = 'simulacion'
urlpatterns = [
    path('', views.home, name='home'),
    path('emprendimientos/', views.lista_emprendimientos, name='lista_emprendimientos'),
    path('simulacion/', views.simulacion, name='simulacion'),
    path('publicaciones/<int:id_emprendimiento>/', views.publicaciones, name='publicaciones'),
    path('comentarios/<int:id_emprendimiento>/', views.comentarios, name='comentarios'),
    path('guardar-publicaciones/', views.guardar_publicaciones, name='guardar_publicaciones'),
    path('predicciones/', views.predicciones, name='predicciones'),
    path('evaluacion/', views.evaluacion, name='evaluacion'),
]