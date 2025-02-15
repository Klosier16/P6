from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Define the model architecture
        self.model = Sequential([
            # Rescaling layer
            Rescaling(1./255, input_shape=input_shape),
            
            # Convolutional + MaxPooling layers
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # Flatten and Fully Connected Layer
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(categories_count, activation='softmax')
        ])
    # Compile the model
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.002),  
            loss='categorical_crossentropy',         
            metrics=['accuracy']                      
        )
