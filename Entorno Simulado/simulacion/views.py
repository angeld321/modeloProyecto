from django.shortcuts import render
from django.http import HttpResponse
from .models import Emprendimiento, Municipio, Tematica, Alcance


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
    return render(request, 'simulacion/simulacion.html')

def predicciones(request):
    return render(request, 'simulacion/predicciones.html')

def evaluacion(request):
    return render(request, 'simulacion/evaluacion.html')