import sys
sys.path.append("../")
from import_libraries import *


class NeuralNetworktf:
    def __init__(self, project, folds, X_train, y_train, X_test, y_test, X_train_all, y_train_all):
        self.project = project
        self.plot_graphs = self.project.plot_graphs
        self.cv_folds = project.cv_folds
        self.folds = folds
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_all = X_train_all
        self.y_train_all = y_train_all

    def tune(self):
        if self.cv_folds > 0 :
            best_model_train, best_model_all = self.tune_model_with_cv(self.folds, self.X_train, self.y_train, self.X_test, self.y_test, self.X_train_all, self.y_train_all)
        else:
            best_model_train, best_model_all = self.tune_model_no_cv(self.X_train, self.y_train, self.X_test, self.y_test, self.X_train_all, self.y_train_all)
        return best_model_train, best_model_all

    def build_model(self, classifier, input_dim):
        """Constructs a Neural Network Model.

        Keyword arguments:
        classifier -- A tensorflow.keras Sequential object
        input_dim -- The input dimension of the model
        """

        # instantiate tensorflow.keras initializer
        weight_initializer = tf.keras.initializers.glorot_uniform()

        # Use the Input layer to specify the input shape
        classifier.add(tf.keras.layers.Input(shape=(input_dim,)))

        # adding the first hidden, second hidden, and output layer
        classifier.add(tf.keras.layers.Dense(units=input_dim, kernel_initializer=weight_initializer, activation="relu"))
        classifier.add(tf.keras.layers.Dense(units=input_dim // 2, kernel_initializer=weight_initializer, activation="relu"))
        classifier.add(tf.keras.layers.Dense(units=1, kernel_initializer=weight_initializer, activation="sigmoid"))
        # keras compile
        classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    # def build_model(self, classifier, input_dim):
    #     """Constructs a Neural Network Model.

    #     Keyword arguments:
    #     classifier -- A tensorflow.keras Sequential object
    #     input_dim -- The input dimension of the model
    #     """

    #     # instantiate tensorflow.keras initializer
    #     weight_initializer = tf.keras.initializers.glorot_uniform()

    #     # adding the input, first hidden, second hidden, and output layer
    #     classifier.add(tf.keras.layers.Dense(units=input_dim // 2, kernel_initializer=weight_initializer,
    #                                             activation="relu", input_dim=input_dim))
    #     classifier.add(tf.keras.layers.Dense(units=input_dim // 2, kernel_initializer=weight_initializer,
    #                                             activation="relu"))
    #     classifier.add(tf.keras.layers.Dense(units=1, kernel_initializer=weight_initializer, activation="sigmoid"))
    #     # keras compile
    #     classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    def finalize_model(self, swa_weights, X_train, y_train, X_test, y_test, X_train_all, y_train_all):
        # build final model
        best_model_train = tf.keras.models.Sequential()
        self.build_model(best_model_train, X_train.shape[1])
        best_model_train.set_weights(swa_weights)

        # fine-tune the average aggregate model using all the data
        all_train_features = np.asarray(X_train).astype(np.float32)
        all_train_labels = np.asarray(y_train).astype(np.float32)
        val_features = np.asarray(X_test).astype(np.float32)
        val_labels = np.asarray(y_test).astype(np.float32)

        history = best_model_train.fit(all_train_features, all_train_labels, validation_data=(val_features, val_labels),
                                        verbose=0, batch_size=16, epochs=6)

        # for use in predict() and other methods
        print(best_model_train.summary())

        if self.plot_graphs:
            self.plot_training_history(history)

        # Refine the model with all data
        X_train_all_val = np.asarray(X_train_all).astype(np.float32)
        y_train_all_val = np.asarray(y_train_all).astype(np.float32)
        best_model_all = tf.keras.models.Sequential()
        self.build_model(best_model_all, X_train_all.shape[1])
        best_model_all.set_weights(swa_weights)
        best_model_all.fit(X_train_all_val, y_train_all_val, verbose=0, batch_size=16, epochs=6)

        return best_model_train, best_model_all

    def tune_model_no_cv(self, X_train, y_train, X_test, y_test, X_train_all, y_train_all):
        minimum_number_of_models = 3
        num_rows_vs_num_models = {
            5000: 10,
            30000: 9,
            50000: 8,
            70000: 7,
            90000: 6,
            150000: 5,
            200000: 4
        }
        # Initialize multiple models to average results - this will depend on how many data points we have
        num_rows = len(X_train)
        num_models = minimum_number_of_models
        for x, y in num_rows_vs_num_models.items():
            if num_rows < x:
                num_models = y
                break

        # Calling clear_session() releases the global state: this helps avoid clutter from old models and layers
        tf.keras.backend.clear_session()
        all_models = [tf.keras.models.Sequential() for _ in range(num_models)]
        swa_weights = []

        # build and fit the models
        for i, keras_model in enumerate(all_models):
            print(f"Training {i + 1}th model")
            # get new subset each time
            train_features, train_labels = self.get_data_subset(X_train, y_train, 0.50)
            # train Model
            self.build_model(keras_model, train_features.shape[1])

            keras_model.fit(train_features, train_labels, validation_data=(X_test, y_test), verbose=0, batch_size=32, epochs=8)
            # keep a running average of the weights
            new_w = keras_model.get_weights()
            if i == 0:
                swa_weights = new_w
            else:
                for i, swa_layer in enumerate(swa_weights):
                    swa_weights[i] = (swa_layer * (i + 1) + new_w[i]) / (i + 2)

        best_model_train, best_model_all = self.finalize_model(swa_weights, X_train, y_train, X_test, y_test, X_train_all, y_train_all)
        return best_model_train, best_model_all

    def tune_model_with_cv(self, folds, X_train, y_train, X_test, y_test, X_train_all, y_train_all):
        tf.keras.backend.clear_session()
        all_models = [tf.keras.models.Sequential() for _ in range(len(folds))]
        swa_weights = []
        # build and fit the models
        for i, keras_model in enumerate(all_models):
            fold = folds[i]
            X_train_val, y_train_val = fold['X_train'], fold['y_train']
            X_test_val, y_test_val = fold['X_test'], fold['y_test']
            self.build_model(keras_model, X_train_val.shape[1])

            X_train_val = np.asarray(X_train_val).astype(np.float32)
            y_train_val = np.asarray(y_train_val).astype(np.float32)
            X_test_val = np.asarray(X_test_val).astype(np.float32)
            y_test_val = np.asarray(y_test_val).astype(np.float32)
            keras_model.fit(X_train_val, y_train_val, validation_data=(X_test_val, y_test_val), verbose=0, batch_size=32, epochs=8)
            new_w = keras_model.get_weights()
            if i == 0:
                swa_weights = new_w
            else:
                for i, swa_layer in enumerate(swa_weights):
                    swa_weights[i] = (swa_layer * (i + 1) + new_w[i]) / (i + 2)

        best_model_train, best_model_all = self.finalize_model(swa_weights, X_train, y_train, X_test, y_test, X_train_all, y_train_all)
        return best_model_train, best_model_all

    def plot_training_history(self, history):
        """
        Plots the training history of the model.

        Keyword arguments:
        history -- the history of the training process of the model.
        """

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(history.history['loss'], "ro", label="Train")
        ax.plot(history.history['val_loss'], "bo", label="Validation")
        ax.set_ylabel("Loss (dots)")
        ax.grid(True)
        ax.set_xlabel("Epochs")
        ax.legend()

        ax1 = ax.twinx()
        ax1.plot(history.history['accuracy'], "r--")
        ax1.plot(history.history['val_accuracy'], "b--")
        ax1.set_ylabel("Accuracy (dashed lines)")

        plt.show()

    def evaluate_model(self, model, X_train, y_train, X_test, y_test) :
        X_train_val = np.asarray(X_train).astype(np.float32)
        y_train_val = np.asarray(y_train).astype(np.float32)
        X_test_val = np.asarray(X_test).astype(np.float32)
        y_test_val = np.asarray(y_test).astype(np.float32)

        # fine-tune the average aggregate model using all the data
        history = model.fit(X_train_val, y_train_val, validation_data=(X_test_val, y_test_val), verbose=0, batch_size=16, epochs=6)
        self.plot_training_history(history)
        # for use in predict() and other methods
        print(model.summary())
