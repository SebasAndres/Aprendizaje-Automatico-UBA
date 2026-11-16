> 8/10

# Regresion Lineal y descenso del gradiente

## Regresión

$f(X) = \sum_{i=0}^p w_i x_i$
$y = f(X) + \epsilon$

## MSE para regresión

$MSE(h, X, Y) = \frac{1}{|X|} \sum_{i=1}^{|X|} (y^{(i)} - h(x^{(i)}))^2$

$MSE(h, X, Y) = \frac{1}{|X|} \sum_{i=1}^{|X|} (y^{(i)} - \sum_{k=0}^p w_k x_k^{(i)}))^2$


# Estimación de parámetros

> Métodos para estimar los mejores parámetros. Minimizar el MSE.
> El MSE garantiza que sea derivable.

$W = \argmin_W MSE(h_W, X, Y)$

Sea $Xd = X$'.

Solución analítica: $W = (Xd_\text{train}^T Xd_\text{train})^{-1} Xd_\text{train}^T y_\text{train}$

## Descenso del gradiente
> 1847, Cauchy

$W^{\text{paso=k}} \leftarrow W^{\text{paso=k-1}} - \alpha \times update (Xd_\text{train}, y_\text{train}, W^{\text{paso=k-1}})$

Es agnóstico a la función de Loss.

~~~py
def cauchy_gradient_descent(
    loss: Betas -> Target,
    d_loss: Betas -> Gradiente, # un array con un numero por cada beta
    initial_betas: Betas,
    learning_rate: float,
    num_iterations: int,
    tol: float = 0.001
):  
    """
    Donde:
        Betas :: Array[float]^p
        Gradiente :: Array[float]^p
        Target :: float
    """

    betas = initial_betas
    for _ in range(num_iterations):
        gradient = d_loss(betas)
        betas_new = betas - learning_rate * gradient

        # hay muchos criterios de parada, este es uno
        if abs(loss(betas_new) - loss(betas)) < tol:
            break
        betas = betas_new    
    return betas
~~~

Por qué dentro del fit se define Loss y d_Loss las funciones (en el ejemplo de las slides):
    i- Errores de tipado.
    ii- La funcion de minimizar no está definida fuera del 
    fit, depende de los datos del dataset.

### Gradient Descent con Mini Batch 
> Solución para datasets grandes (no entra en memoria)

- Si batch_size = n es batch gradient descent (el que ya vimos),
- Si 1 < batch_size < n, es mini-batch GD,
- Si batch_size = 1 --> stochastic GD.

~~~py
miniBatchGD(X, Y, lr, beta_init, loss, d_loss):
    beta = beta_init 

    not_used = (X, Y)
    while not stop:        
        while not not_used.empty():        
            X_batch, Y_batch = muestrear_sin_reposicion(not_used)

            grad = d_loss(X_batch, Y_batch, betas)
            betas = betas - lr * grad

            stop = _check_stop_criteria(*)            
    return beta
~~~

Acá dejamos de minimizar el Loss(X_train, Y_train), minimizamos algo que se le asemeja idealmente. 

----
**Iteración / step**: Es un ciclo de iteración de los parámetros del modelo utilizando un mini-batch específico.

**Época / epoch**: Es un ciclo completo de entrenamiento (todos los datos). Una epoca tiene muchas iteraciones, tantas como cantidad de mini-batches que conforman el Dataset.

# Pytorch

En pytorch las funciones están preparadas para trabajar con minibatches.

~~~python
import torch

class RegL(torch.nn.Module):
    def __init__(self, dim_features):
        super().__init__()
        self.W = torch.nn.Parameter(data=torch.rand(dim_features,1))
        self.w0 = torch.nn.Parameter(data=torch.zeros(1))

    def forward(self, x_batch):
        # x_batch :: (batch_size x dim_features)
        # W :: (dim_features x 1)
        return (x_batch @ self.W) + self.w0 


def train(
    model, 
    dataloader_train, 
    dataloader_val,
    n_epochs,
    lr
):
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.params, lr=lr)

    for epoch in epochs:
        for x_batch, y_batch in dataloader_train:
            y_pred = model(x_train_batch)
            loss = mse(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backwards() # calcula los gradientes de loss respecto a todos los params
            optimizer.step() # actualiza parametros en la direccion que reduce la pérdida

        # Monitoreo
        with torch.no_grad():
            for x_batch, y_batch in dataloader_val:
                y_pred = model(x_train_batch)
                loss = mse(y_pred, y_batch)
                print(epoch, loss)
~~~

### Regularización y funciones base

> Seguro no se evalúa

**Funciones base**: Son funciones que transforman los atributos de entrada para que un modelo lineal pueda representar relaciones no lineales. Análogo a la idea del kernel $X' \leftarrow \phi(X)$. Ej: `PolyFeatures(*)`.

Muchas veces es necesario restringir los pesos que un algoritmo aprende. Más features puede causar overfitting. Queremos focalizar el efecto / modelo en pocos atributos. Además evitar problemas no invertibles por colinearidades.

**Regularización**:
Agregamos un nuevo término en las funciones de costo.... 

Opciones:

$J_\text{Ridge} = MSE_{X, y}(w) + \lambda ||w||_2^2$

$J_\text{Lasso} = MSE_{X, y}(w) + \lambda ||w||_1$

$J_\text{ElasticNet} = MSE_{X, y}(w) + \lambda_1 ||w||_1 + \lambda_2 \lambda ||w||_2^2$


**Observaciones:**
- $w_0$ no se regulariza
- Si lambda crece --> w tiende a 0.
- Lasso tiende a dar $W$ óptimos con algunos $w_i = 0$ ($\therefore$ menos features usadas). En Ridge, cercano a 0 pero no 0 directamente.
- Antes de aplicar regularización hay que escalar los features por lo general.
