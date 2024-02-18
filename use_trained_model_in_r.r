plot(1:10)


library(keras)
library(tensorflow)


# Loading a trained model
# Loading the model
loaded_model = tf$keras$models$load_model(
    'triangle_test.keras', 
    custom_objects={
        'LinearMasked': LinearMasked,
        'custom_loss': dag_loss
    }
)