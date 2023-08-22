#import libraries
# Importación de las librerías
import numpy as np
import pandas as pd
import datetime 
import modulo_conn_sql as mcq
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pickle import dump
from pickle import load


################ RUN NEURAL NETWORK #################################3
# Metodos auxiliares
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

def get_dataset(
    procedure_name: str,
    param_pais: str,
    param_fecha_inicio: datetime,
    param_fecha_fin: datetime ) -> pd.core.frame.DataFrame:
    
    df = querySQL( 
        "{CALL " + procedure_name+ " (?,?,?)}", 
        ( 
            param_pais, 
            param_fecha_inicio.strftime("%Y-%m-%d"), 
            param_fecha_fin.strftime("%Y-%m-%d") 
        ) 
    )
    
    return df



def split_train_test(
    param_df: pd.core.frame.DataFrame, 
    date_split:str ) -> (list, list):
    
    # Convertir dataframe a lista de listas, solo se toman las columnas desde el volumen
    dataset = param_df.iloc[: , 1:].values

    # definicion del conjunto de entrenamiento
    dataset_train = df[ df['Fecha'] < date_split ]
    dataset_train = dataset_train.iloc[:,2:].values

    # definicion del conjunto de test
    dataset_test = df[ df['Fecha'] >= date_split ]
    dataset_test = dataset_test.iloc[:,2:].values
    
    return (dataset_train, dataset_test)
    
def scale_train_test(param_df, param_train, param_test ):
     
    dataset = param_df.iloc[: , 2:].values
    # Definicion de variable para escalar los datos entre 0 y 1
    sc = MinMaxScaler(feature_range = (0, 1))
    dataset = sc.fit(dataset)

    # Ajuste de los datos segun la variable escaladora
    dataset_train = sc.transform(param_train)
    dataset_test = sc.transform(param_test)
    
    # save the scaler
    dump(sc, open("../datos/" + pais + "/" + pais +'.pkl', 'wb'))
    
    return (dataset_train, dataset_test) 

def train_model(
    param_dataset: list,
    timesteps : int,
    layers : int,
    units: int,
    dropout: float, 
    epochs: int, 
    batch: int,
    pais: str
)-> None:
    
    predictor_variables_num = param_dataset.shape[1] #obtener numero de variables 
    
    #define array by predictor variable
    variables = [ [] for i in range(0, predictor_variables_num) ]
    X_train = []
    y_train = []

    for i in range(timesteps, param_dataset.shape[0] ):
        
        # process for each variable
        for j in range(0, predictor_variables_num):
            
            variables[j].append( param_dataset[i-timesteps:i, j] )
        
        y_train.append(param_dataset[i, 0])
    
    #convert to numpy objects
    for i in range(0, predictor_variables_num):
        variables[i] = np.array(variables[i])
    y_train = np.array(y_train)
    
    #reshape numpy objects
    for i in range(0, predictor_variables_num):
        variables[i] = np.reshape(variables[i], (variables[i].shape[0], variables[i].shape[1], 1 ))
        
    #build tensor structure for LSTM
    
    # if just one variable
    if predictor_variables_num == 1:
        X_train = variables[0]
    
    else:
        X_train = np.append(variables[0], (variables[1]), axis=2)
    
    # append to x_train if more than 2 variables
    if predictor_variables_num > 2:
        for i in range(2, predictor_variables_num):

                X_train = np.append(X_train, (variables[i]), axis=2)
    
    #inizializate regressor
    regressor = Sequential()
    
    #if there is more than a layer, return input dimension to the next layer, through return_sequence parameter
    rs = True if layers > 1  else False
    
    print(X_train.shape)
    print(X_train.shape[1])
    print(X_train.shape[2])
    for i in range( 0, layers):

        # if the first layer, define input dimensions
        if i == 0:
            regressor.add(LSTM(units=units, return_sequences = rs, input_shape = (X_train.shape[1], X_train.shape[2])))
        else:
            regressor.add(LSTM(units=units, return_sequences = rs))
            
        #if dropout layers
        if dropout > 0.0:
            regressor.add(Dropout(dropout))
            
        # penultimate layer dont return input dimensions, because the last one just has 1 neuron
        if i == layers -2: 
            rs = False
            
    #output layer
    regressor.add(Dense(units=1))
    
    #compile RNR
    regressor.compile(optimizer = 'adam', loss='mean_squared_error')
        
    ##regressor.summary()
    
    #fit the RNR to datatrain
    regressor.fit(X_train, y_train, 
                  epochs=epochs, 
                  batch_size = batch )
    
    #save model
    regressor.save("../datos/" + pais +"/" + pais + "_testing.h5")
    
def test_model(
    param_train : list, 
    param_test: list, 
    test_no_scale : list,
    timesteps: int,
    pais: str ):
    
    predictor_variables_num = param_train.shape[1] #obtener cantidad de elementos dentro de cada dimension
    
    # load regressor in testing
    regressor_test = load_model("../datos/" + pais +"/" + pais + "_testing.h5")
    
    # load the scaler
    scaler = load(open("../datos/" + pais + "/" + pais + '.pkl', 'rb'))
    
    #inputs are last timesteps for first prediction day
    inputs = param_train[len(param_train) - timesteps: ]
    
    #this process will be excecute for every prediction day
    for j in range (0, param_test.shape[0] ):
        
        X_test = []
        #define array by predictor variable
        variables = [ [] for i in range(0, predictor_variables_num) ]
        
        for i in range(timesteps, inputs.shape[0]+1 ):
        
            # process for each variable
            for k in range(0, predictor_variables_num):

                variables[k].append( inputs[i-timesteps:i, k] )
                
        #convert to numpy objects
        for i in range(0, predictor_variables_num):
            variables[i] = np.array(variables[i])

        #reshape numpy objects
        for i in range(0, predictor_variables_num):
            variables[i] = np.reshape(variables[i], (variables[i].shape[0], variables[i].shape[1], 1 ))
        
             
        #build tensor structure for LSTM
        # if just one variable
        if predictor_variables_num == 1:
            X_test = variables[0]

        else:
            X_test = np.append(variables[0], (variables[1]), axis=2)

        # append to x_test if more than 2 variables
        if predictor_variables_num > 2:
            for i in range(2, predictor_variables_num):

                X_test = np.append(X_test, (variables[i]), axis=2)

        #make prediction
        prediction = regressor_test.predict(X_test)
        
        #to prediction append another regressor variables
        prediction = np.append(prediction, (param_test[:len(prediction), 1 : ]), axis=1 )

        inputs = param_train[len(param_train) - timesteps:]

        inputs = np.append(inputs, (prediction), axis=0 )

  
    print("MSE Test: "+str(mean_squared_error(prediction,param_test)))
    
    prediction = scaler.inverse_transform(prediction)
    
    return prediction
    
 
def forecast(
    df_forecast: pd.core.frame.DataFrame,
    start_date_forecast : datetime ):

    dataset_pred = df_forecast[['DiaSemana','DiaSemana', 'Mes', 'Semana_Relativa', 'Semanas_mes', 'Año']]

    #convert to list
    predict_set = dataset_pred.iloc[:,:].values

    # load the scaler
    scaler = load(open("../datos/" + pais + "/" + pais + '.pkl', 'rb'))

    # load regressor in testing
    regressor_test = load_model("../datos/" + pais +"/" + pais + "_testing.h5")

    #scale forecast dataset 
    predict_set_scaled = scaler.transform(predict_set)

    #define input dataset to start forecast
    dataset_test = df[ df['Fecha'] < start_date_forecast ]
    test_set  = dataset_test.iloc[:, 2:].values
    # scale features
    test_set_scaled = scaler.transform(test_set)

    #los inputs van a ser los ultimos Timesteps dias del training set, para predecir el primer dia 
    inputs = test_set_scaled[len(test_set_scaled) - timesteps: ]

    predictor_variables_num = predict_set_scaled.shape[1] 

    #this process will be excecute for every prediction day
    for j in range (0, predict_set_scaled.shape[0] ):
        X_test = []
        #define array by predictor variable
        variables = [ [] for i in range(0, predictor_variables_num) ]

        for i in range(timesteps, inputs.shape[0]+1 ):

                # process for each variable
                for k in range(0, predictor_variables_num):

                    variables[k].append( inputs[i-timesteps:i, k] )
        #convert to numpy objects
        for i in range(0, predictor_variables_num):
            variables[i] = np.array(variables[i])

        #reshape numpy objects
        for i in range(0, predictor_variables_num):
            variables[i] = np.reshape(variables[i], (variables[i].shape[0], variables[i].shape[1], 1 ))

        #build tensor structure for LSTM
        # if just one variable
        if predictor_variables_num == 1:
            X_test = variables[0]

        else:
            X_test = np.append(variables[0], (variables[1]), axis=2)

        #append to x_test if more than 2 variables
        if predictor_variables_num > 2:
            for i in range(2, predictor_variables_num):

                X_test = np.append(X_test, (variables[i]), axis=2)     

        #make prediction
        prediction = regressor_test.predict(X_test)

        #to prediction append another regressor variables
        prediction = np.append(prediction, (predict_set_scaled[:len(prediction), 1 : ]), axis=1 )

        inputs = test_set_scaled[len(test_set_scaled) - timesteps:]

        inputs = np.append(inputs, (prediction), axis=0 )

    prediction = scaler.inverse_transform(prediction)
    
    return prediction

#Run parameters for neural network
#Parametros SQL
pais = 'Republica Dominicana'  
inicioHistoria = datetime.datetime(2021,1, 1) #'2013-05-01'
finHistoria = datetime.datetime.today() #fecha actual
fecha_split = '2023-5-01'

df = get_dataset("SCAC_AP4_Serie_VolumenDiario", pais, inicioHistoria, finHistoria)

#SPLIT TRAIN - TEST
dataset_train_nscale, dataset_test_nscale = split_train_test(df, fecha_split) 

#SCALE VARIABLES
dataset_train, dataset_test = scale_train_test(df, dataset_train_nscale,dataset_test_nscale ) 

print("dataset_train--------------")
print(dataset_train)
print(dataset_train.shape)

#TRAIN MODEL

timesteps = 15
layers = 5
units = 30
dropout = 0.2
epochs = 5
batch = 8

train_model(dataset_train, timesteps, layers, units, dropout, epochs, batch, pais)

#Parameters
start_date_forecast = datetime.datetime(2023, 6, 1)
end_date_forecast = datetime.datetime(2023, 6, 30)

df2 = get_dataset("SCAC_AP4_Serie_VolumenDiario_AuxFecha", pais, start_date_forecast, end_date_forecast)

result = forecast(df2, start_date_forecast)

#mensual results
df_result = pd.DataFrame({'Forecast':result[:, 0]})
df_result = pd.concat([df2, df_result], axis=1)
df_result.groupby(['Año','Mes'])['Forecast'].sum()


        