from django.shortcuts import render, redirect
from django.db.models import Count
from django.http import JsonResponse
from .models import Emprendimiento, Municipio, Alcance, Tematica, Publicacion, Seguidores, Comentario
import random
import json
import requests  # Añadí requests
import os
from dotenv import load_dotenv
from django.shortcuts import render, redirect, get_object_or_404




def home(request):
    return render(request, 'simulacion/home.html')

def lista_emprendimientos(request):
    # Obtener filtros desde los parámetros GET
    municipio_id = request.GET.get('municipio')
    alcance_id = request.GET.get('alcance')
    tematica_id = request.GET.get('tematica')
    sort_by = request.GET.get('sort_by', 'id_emprendimiento')  # Por defecto, ordenar por ID
    sort_order = request.GET.get('sort_order', 'asc')  # Por defecto, ascendente

    # Consulta base con optimización
    emprendimientos = Emprendimiento.objects.all().select_related('id_municipio_origen', 'id_alcance').prefetch_related('tematicas')

    # Aplicar filtros si existen
    if municipio_id:
        emprendimientos = emprendimientos.filter(id_municipio_origen__id_municipio=municipio_id)
    if alcance_id:
        emprendimientos = emprendimientos.filter(id_alcance__id_alcance=alcance_id)
    if tematica_id:
        emprendimientos = emprendimientos.filter(tematicas__id_tematica=tematica_id)

    # Aplicar ordenación
    if sort_by == 'id_emprendimiento':
        if sort_order == 'desc':
            emprendimientos = emprendimientos.order_by('-id_emprendimiento')
        else:
            emprendimientos = emprendimientos.order_by('id_emprendimiento')
    elif sort_by == 'municipio':
        if sort_order == 'desc':
            emprendimientos = emprendimientos.order_by('-id_municipio_origen__municipio')
        else:
            emprendimientos = emprendimientos.order_by('id_municipio_origen__municipio')

    # Obtener opciones para los filtros
    municipios = Municipio.objects.all()
    alcances = Alcance.objects.all()
    tematicas = Tematica.objects.all()

    context = {
        'emprendimientos': emprendimientos,
        'municipios': municipios,
        'alcances': alcances,
        'tematicas': tematicas,
        'sort_by': sort_by,
        'sort_order': sort_order,
    }
    return render(request, 'simulacion/emprendimientos.html', context)

def simulacion(request):
    # Obtener todos los emprendimientos con conteos de publicaciones y comentarios
    emprendimientos = Emprendimiento.objects.all().select_related('id_municipio_origen', 'id_alcance').prefetch_related('tematicas').annotate(
        num_publicaciones=Count('publicacion', distinct=True),
        num_comentarios=Count('publicacion__comentario', distinct=True)
    )
    context = {
        'emprendimientos': emprendimientos,
    }
    return render(request, 'simulacion/simulacion.html', context)



def guardar_publicaciones(request):
    if request.method == 'POST':
        id_emprendimiento = request.POST.get('id_emprendimiento')
        publicaciones_texto = request.POST.get('publicaciones', '')
        
        try:
            emprendimiento = Emprendimiento.objects.get(id_emprendimiento=id_emprendimiento)
            seguidores = Seguidores.objects.filter(id_emprendimiento=emprendimiento).first()
            num_seguidores = seguidores.cantidad if seguidores else 0

            # Procesar el texto para extraer las publicaciones
            publicaciones = [pub.strip() for pub in publicaciones_texto.split('),')]
            publicaciones = [pub.replace('(', '').replace(')', '').strip() for pub in publicaciones if pub.strip()]
            
            for contenido in publicaciones:
                # Calcular likes basados en seguidores
                caracteres = len(contenido)
                if num_seguidores == 0:
                    n_likes = random.randint(0, 5)
                else:
                    max_likes = min(num_seguidores // 10, 100)
                    n_likes = random.randint(0, max_likes)
                
                Publicacion.objects.create(
                    contenido=contenido,
                    n_likes=n_likes,
                    id_emprendimiento=emprendimiento
                )

            return redirect('simulacion:comentarios', id_emprendimiento=id_emprendimiento)
        except Emprendimiento.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Emprendimiento no encontrado'}, status=404)
    
    return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)

def ver_publicaciones(request):
    if request.method == 'POST':
        id_emprendimiento = request.POST.get('id_emprendimiento')
        try:
            emprendimiento = get_object_or_404(Emprendimiento, id_emprendimiento=id_emprendimiento)
            publicaciones = Publicacion.objects.filter(id_emprendimiento=emprendimiento).annotate(
                num_comentarios=Count('comentario')
            ).values('id_publicacion', 'contenido', 'num_comentarios', 'n_likes')
            
            return JsonResponse({
                'status': 'success',
                'publicaciones': list(publicaciones),
                'emprendimiento': {
                    'id': emprendimiento.id_emprendimiento,
                    'nombre': emprendimiento.nombre_emprendimiento
                }
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)

def publicaciones(request, id_emprendimiento):
    emprendimiento = get_object_or_404(Emprendimiento, id_emprendimiento=id_emprendimiento)
    return render(request, 'simulacion/publicaciones.html', {
        'emprendimiento': emprendimiento
    })

def comentarios(request, id_emprendimiento):
    emprendimiento = get_object_or_404(Emprendimiento, id_emprendimiento=id_emprendimiento)
    publicaciones = Publicacion.objects.filter(id_emprendimiento=emprendimiento).annotate(
        num_comentarios=Count('comentario')
    )
    return render(request, 'simulacion/comentarios.html', {
        'emprendimiento': emprendimiento,
        'publicaciones': publicaciones
    })

def predicciones(request):
    return render(request, 'simulacion/predicciones.html')

def evaluacion(request):
    return render(request, 'simulacion/evaluacion.html')