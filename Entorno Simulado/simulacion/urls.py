from django.urls import path
from . import views

app_name = 'simulacion'
urlpatterns = [
    path('', views.home, name='home'),
    path('emprendimientos/', views.lista_emprendimientos, name='lista_emprendimientos'),
    path('emprendimientos/agregar/', views.agregar_emprendimiento, name='agregar_emprendimiento'),
    path('emprendimientos/modificar/', views.modificar_emprendimiento, name='modificar_emprendimiento'),
    path('simulacion/', views.simulacion, name='simulacion'),
    path('publicaciones/<int:id_emprendimiento>/', views.publicaciones, name='publicaciones'),
    path('comentarios/<int:id_emprendimiento>/', views.comentarios, name='comentarios'),
    path('guardar-publicaciones/', views.guardar_publicaciones, name='guardar_publicaciones'),
    path('agregar-comentarios/<int:id_publicacion>/', views.agregar_comentarios, name='agregar_comentarios'),
    path('guardar-comentarios/', views.guardar_comentarios, name='guardar_comentarios'),
    path('ver-comentarios/<int:id_publicacion>/', views.ver_comentarios, name='ver_comentarios'),
    path('predicciones/', views.predicciones, name='predicciones'),
    path('predicciones_data/', views.predicciones_data, name='predicciones_data'),
    path('recommend_emprendimientos/', views.recommend_emprendimientos, name='recommend_emprendimientos'),
    path('generate_pdf_report/', views.generate_pdf_report, name='generate_pdf_report'),    
    path('evaluacion/', views.evaluacion, name='evaluacion'),    
    path("evaluacion/data/", views.evaluacion_data, name="evaluacion_data"),
    path("evaluacion/recalculate/", views.recalculate_metrics, name="recalculate_metrics"),
    path("evaluacion/model_summary/", views.model_summary, name="model_summary"),
    path('check_missing_embeddings/', views.check_missing_embeddings, name='check_missing_embeddings'),
    path('generar_embeddings/', views.generar_embeddings, name='generar_embeddings'),
]