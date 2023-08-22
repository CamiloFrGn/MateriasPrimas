# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:36:22 2022

@author: jsdelgadoc
"""

import modulo_conn_sql as mcq
import numpy as np
import pandas as pd 
import datetime 
from scipy import stats

import sqlalchemy as sa
import urllib

from io import BytesIO
from flask import send_file

def conectarSQL():
    conn = mcq.ConexionSQL()
    cursor = conn.getCursor()
    return cursor

#Query BD SQL-Server Cemex
def querySQL(query, parametros):
    #Conectar con base sql y ejecutar consulta
    cursor = conectarSQL()
    try:
        cursor.execute(query, parametros)
        #obtener nombre de columnas
        names = [ x[0] for x in cursor.description]
        
        #Reunir todos los resultado en rows
        rows = cursor.fetchall()
        resultadoSQL = []
            
        #Hacer un array con los resultados
        while rows:
            resultadoSQL.append(rows)
            if cursor.nextset():
                rows = cursor.fetchall()
            else:
                rows = None
                
        #Redimensionar el array para que quede en dos dimensiones
        resultadoSQL = np.array(resultadoSQL)
        resultadoSQL = np.reshape(resultadoSQL, (resultadoSQL.shape[1], resultadoSQL.shape[2]) )
    finally:
            if cursor is not None:
                cursor.close()
    return pd.DataFrame(resultadoSQL, columns = names)

def gen_muestra(parametros_dist, distribucio_param, num_muestras):
    
    muestras = distribucio_param.rvs(*parametros_dist, size=num_muestras)
    
    muestras =  [abs(ele) for ele in muestras]
        
    return sum(muestras)

def materias_primas_programacion( fecha_param, pais_param):
    
    #Dataset del forecast
    df_programacion = querySQL( "SELECT * FROM SCAC_AT15_Programacion7Dias WHERE FechaEntrega = ?" ,(fecha_param))
    df_programacion['Posicion'] = df_programacion['Posicion'].astype(float)
    df_programacion['VolPartida'] = df_programacion['VolPartida'].astype(float)

    #agrego informacion geografica        
    nombre_cluster = querySQL( "SELECT Pais, Centro, Ciudad_Cluster as Ciudad, [Desc Cluster] as Cluster, [Planta Unica] as PlantaUnica FROM SCAC_AT1_NombreCluster where Pais = ? and Activo = 1" , (pais_param) )
    
    #materiales a desagregar
    lista_materiales = ['ADITIVO', 'ARENA', 'Agua', 'CEMENTO', 'CENIZA', 'GRAVA']

    estatus = ['Confirmada - Cabecera', 
               'Confirmada - Cabecera', 
               'En proceso - Cabecera',
               'Por confirmar - Cabecera',
               'Bloqueada - Cabecera']

  
    #Dataset desagregacion materias primas
    param_dist = querySQL( "SELECT * FROM SCAC_AT42_parametros_distribucion_materiales" , () )
    iteraciones= 1

    # Definición de la distribución
    distribucion = stats.johnsonsu
    df_materiales = pd.DataFrame()

    for j in range(0, iteraciones):
       # print("iteracion: " + str(j))

        df_programacion_dia = pd.merge(df_programacion, nombre_cluster[['Centro', 'PlantaUnica', 'Cluster']], left_on='Planta', right_on='Centro' )
        df_programacion_dia = df_programacion_dia[ (df_programacion_dia['Posicion'] < 3000) & 
                                                 (df_programacion_dia['EstatusPedido'].isin(estatus) ) ][['Cluster','PlantaUnica', 'EstatusPedido', 'VolPartida']]\
                                                 .groupby(['Cluster', 'PlantaUnica', 'EstatusPedido'])['VolPartida'].sum().reset_index()

        iteraciones = 10
        lista_plantas = df_programacion_dia['PlantaUnica'].unique()
        
        for k in range(0, iteraciones):
            
            for i in lista_plantas:
                #print(i)
                if(len(df_programacion_dia) > 0 ):
                    #ciclo para generar muestras por cada material
                    df_programacion_dia_parcial = df_programacion_dia[df_programacion_dia['PlantaUnica'] == i] #param_dist[(param_dist['ubicacion']==i) & (param_dist['material']==lista_materiales[j])]
    
                    for j in range(0, len(lista_materiales)):
                        #print(lista_materiales[j])
                        parametros_1 = param_dist[(param_dist['ubicacion']==i) & (param_dist['material']==lista_materiales[j])][['param1','param2','param3','param4']].values
                        #print(parametros_1)
    
                        if (len(parametros_1) > 0):
                            df_programacion_dia_parcial[lista_materiales[j]] = df_programacion_dia_parcial.apply(lambda x: gen_muestra(tuple([float(h) for h in parametros_1[0]]), distribucion, int(x['VolPartida'])), axis=1)
    
                        else : 
                            df_programacion_dia_parcial[lista_materiales[j]] = 0
    
                    if len(df_materiales) == 0:
                        df_materiales = df_programacion_dia_parcial
                    else: 
                        df_materiales = pd.concat([df_materiales, df_programacion_dia_parcial])

    df_materiales = df_materiales.groupby(['Cluster', 'PlantaUnica', 'EstatusPedido', 'VolPartida'])['ADITIVO','ARENA','Agua','CEMENTO','CENIZA','GRAVA'].median().reset_index()
    
    df_materiales = df_materiales.fillna(0)
    
    #ajuste de unidades
    df_materiales['ADITIVO'] = (df_materiales['ADITIVO']/1000.0).astype(int)
    df_materiales['ARENA'] = (df_materiales['ARENA'] / 1000.0).astype(int)
    df_materiales['CEMENTO'] = (df_materiales['CEMENTO'] / 1000.0).astype(int)
    df_materiales['CENIZA'] = (df_materiales['CENIZA'] / 1000.0).astype(int)
    df_materiales['GRAVA'] = (df_materiales['GRAVA'] / 1000.0).astype(int)
    df_materiales['Agua'] = (df_materiales['Agua']).astype(int)
    
    #agregar medidas a los nombres de columnas
    df_materiales.rename(columns = {'VolPartida': 'Concreto (m3)' ,'ADITIVO':'ADITIVO (L)', 'ARENA':'ARENA (TN)', 'Agua' : 'Agua (L)', 'CEMENTO':'CEMENTO (TN)', 'CENIZA': 'CENIZA (TN)', 'GRAVA':'GRAVA (TN)' }, inplace = True)
    
    return df_materiales[['Cluster', 'PlantaUnica', 'EstatusPedido', 'Concreto (m3)', 'ADITIVO (L)', 'ARENA (TN)', 'CEMENTO (TN)', 'GRAVA (TN)' ]]

def exportar_materias_primas_programacion(fecha, pais):
    df_prog = materias_primas_programacion(fecha, pais)
    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df_prog.to_excel( writer, sheet_name="MMPP" )
    writer.close()
    output.seek(0)
    return send_file(output, attachment_filename="MMPP_" + pais + "_" + fecha + "_" +pd.to_datetime("now").strftime("%Y%m%d%H%M%S") + ".xlsx", as_attachment=True)
    
    