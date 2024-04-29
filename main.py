import numpy as np
from mnist import MNIST
import tkinter as tk
from time import perf_counter


# parent class for neural network models
class NeuralNetwork:
    def __init__(self) -> None:
        # input and output values 
        self.input: np.array = None
        self.output: np.array = None

        # training variables
        self.hidden_nodes: int = 0
        self.epochs: int = 0
        self.learning_rate: int = 0

        # weights and biases
        self.output_weights: np.array = None
        self.hidden_weights: np.array = None
        self.hidden_biases: np.array = None
        self.output_biases: np.array = None

        # error values
        self.output_error: np.array = None
        self.hidden_error: np.array = None

        # output values
        self.hidden_output: np.array = None
        self.prediction: np.array = None

    # for getting and normalising the MNIST dataset
    @staticmethod
    def get_dataset(training: bool) -> np.array:
        # load MNIST dataset
        try:
            dataset = MNIST('dataset')

            # get either training or testing data
            if training:
                images, labels = dataset.load_training()
            else:
                images, labels = dataset.load_testing()

            # return normalised data
            return np.array(images) / 255.0, np.eye(10)[np.array(labels)]

        # file not found error handling
        except FileNotFoundError:
            data_type = "Training" if training else "Testing"
            print(f"Error: MNIST {data_type} Data files not found")
            exit()

    # sigmoid function, compresses number-line to values between 0 - 1
    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    # softmax function, provides probability distribution of computed results
    @staticmethod
    def softmax(x: np.array) -> np.array:
        return np.exp(x) / np.sum(np.exp(x))

    # base loss function, mean squared error
    def loss_function(self) -> float:
        return np.mean((self.output - self.prediction)**2)

    # set initial weights and biases
    def training_values(self, input_size: int, output_size: int):
        # setting randomised weights using bell curve distribution
        self.hidden_weights = np.random.randn(input_size, self.hidden_nodes)
        self.output_weights = np.random.randn(self.hidden_nodes, output_size)

        # sets all biases to zero
        self.hidden_biases = np.zeros(self.hidden_nodes)
        self.output_biases = np.zeros(output_size)

    # load values of previously trained parameters, .npz files
    def import_values(self, name: str):
        try:
            with (np.load(f'trained_parameters\\{name}.npz' if name != '' else 'trdata.npz', allow_pickle=True)
                  as loaded_data):
                self.hidden_weights = loaded_data['hidden_weights']
                self.output_weights = loaded_data['output_weights']
                self.hidden_biases = loaded_data['hidden_biases']
                self.output_biases = loaded_data['output_biases']

        except FileNotFoundError:
            print("Error: File not found")
            # asks user if they want to try import file again, recursive call
            if (input("Retry File Import: ").upper()).startswith("Y"):
                self.import_values(input('File Name: '))

    # export values of trained network as .npz
    def export_values(self, name: str):
        # standard convention name
        standard_name = f"{self.hidden_nodes}h{self.epochs}e{self.learning_rate:.2e}l"
        try:
            np.savez(f'trained_parameters\\{name}' if name != '' else standard_name,
                     hidden_weights=self.hidden_weights,
                     output_weights=self.output_weights,
                     hidden_biases=self.hidden_biases,
                     output_biases=self.output_biases)

        except FileExistsError:
            print("Error: File already exists")
            # asks user if they want to try import file again, recursive call
            if (input("Retry File Export: ").upper()).startswith("Y"):
                self.export_values(input('File Name: '))


# class for the feed forward model
class FeedForwardModel(NeuralNetwork):
    def __init__(self) -> None:
        super().__init__()

    # cross entropy loss, used to calculate neural network accuracy
    def loss_function(self) -> float:
        return -np.sum(self.output * np.log(self.prediction + 1.0e-10))  # small value to avoid log(0)

    # feed data forward through the neural network
    def feed_forward(self):
        # calculate hidden layer output, sigmoid
        self.hidden_output = (np.dot(self.input, self.hidden_weights)
                              + self.hidden_biases)  # calculate product of input * weights + bias
        self.hidden_output = self.sigmoid(self.hidden_output)  # sigmoid calculate values

        # calculate output layer final prediction, softmax
        self.prediction = (np.dot(self.hidden_output, self.output_weights)
                           + self.output_biases)  # calculate product of input * weights + bias
        self.prediction = self.softmax(self.prediction)

    # calculate error in hidden and output layers
    def back_propagation(self):
        # calculate error in hidden and output layers
        self.output_error = self.prediction - self.output  # difference between prediction and expected
        self.hidden_error = np.dot(self.output_error, self.output_weights.T) * self.hidden_output * (
                    1 - self.hidden_output)

    # update weights and biases of the neural network
    def update_parameters(self):
        # update weights, current weights minus outer product of input and error
        self.output_weights -= self.learning_rate * np.outer(self.hidden_output, self.output_error)
        self.hidden_weights -= self.learning_rate * np.outer(self.input, self.hidden_error)

        # update biases, biases minus portion of error
        self.output_biases -= self.learning_rate * self.output_error
        self.hidden_biases -= self.learning_rate * self.hidden_error

    # train the neural network on the provided data set (MNIST)
    def train(self):
        # get user to select init training values
        self.hidden_nodes = get_numeric_input("Hidden Nodes", 10, 1, 100)
        self.epochs = get_numeric_input("Epochs", 10, 1, 1000)
        self.learning_rate = get_numeric_input("Learning Rate", 0.01, 0, 1, False)

        # print training values
        print("="*66)
        print(f"Hidden Nodes: {self.hidden_nodes}, Epochs: {self.epochs}, Learning Rate: {self.learning_rate}")
        print("="*66)

        # get MNIST dataset
        t1 = perf_counter()
        images, labels = self.get_dataset(True)
        print(f"Execution Time (dataset extraction): {perf_counter() - t1:.4f}s")

        # timer for getting training speed
        t2 = perf_counter()

        # set initial training values
        self.training_values(len(images[0]), len(labels[0]))

        # for timing the training process, for debugging
        time_total = 0

        print("====================== Training Information ======================")

        # for storing loss values over time
        loss_log = []

        # train with entire dataset epoch amount of times
        for epoch in range(self.epochs):
            # timer for getting epoch train rate
            t3 = perf_counter()

            for image, label in zip(images, labels):
                # set the current input and expected output data
                self.input = image
                self.output = label

                # main training loop
                self.feed_forward()
                self.back_propagation()
                self.update_parameters()

            # get current loss
            loss_log.append(self.loss_function())

            # calculate time
            epoch_time = perf_counter() - t3
            time_total += epoch_time

            # print training information for given epoch
            print(f"Epoch: {epoch + 1}/{self.epochs}, Time: {epoch_time:.4f}s, Loss: {loss_log[epoch]:.10f}")

        # print the time taken to train
        print("="*66)
        print(
            f"Epoch: {self.epochs}/{self.epochs}, Average Time: {time_total / self.epochs:.4f}s,"
            f" Final loss: {loss_log[self.epochs - 1]:.10f}")
        print("="*66)
        print(f"Execution time (training): {perf_counter() - t2:.4f}s")
        print("="*66)

        # get neural network prediction of input data

    def predict(self, input_matrix: np.array, output: np.array, print_result: bool) -> bool:
        # time prediction speed
        t = perf_counter()

        # set input and expected output
        self.input = input_matrix
        self.output = output

        # get the prediction
        self.feed_forward()

        # find the highest probability prediction
        predicted = np.argmax(self.prediction)
        confidence = self.prediction[predicted]

        # print testing information
        if print_result:
            print(
                f"Predicted: {predicted}, Real Value: {np.argmax(output)}, Confidence: {confidence * 100:.2f}%,"
                f" Execution Time: {perf_counter() - t:.4f}s")

        # return true if prediction was correct
        if predicted == np.argmax(output):
            return True
        return False


# drawing grid UI elements
class DrawingGrid:
    def __init__(self, rows, cols, cell_size, prompt, time_limit, player) -> None:
        # main display variables
        self.rows: int = rows
        self.cols: int = cols
        self.cell_size: int = cell_size
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]

        # create the main window
        self.root = tk.Tk()
        self.root.title(f"Drawing Grid, Player {player}")

        # create the canvas for drawing
        self.canvas = tk.Canvas(self.root, width=cols * cell_size, height=rows * cell_size, bg="white")
        self.canvas.pack()

        # bind the mouse click event to draw pixels
        self.canvas.bind("<B1-Motion>", self.draw_pixel)

        # labeling the widget
        self.text_label = tk.Label(self.root, text=f"Player {player}, Prompt: {prompt}", font=("Arial", 24))
        self.text_label.place(x=135, y=10)

        # set time limit
        self.time: int = time_limit
        self.drawing_time: bool = False

        # timer label display
        self.timer_label = tk.Label(self.root, text=f"Drawing Time: {self.time} seconds", font=("Arial", 12))
        self.timer_label.place(x=180, y=60)

    # update the timer and export the matrix when time runs out
    def update_timer(self):
        # update the timer
        self.time -= 1
        self.timer_label.config(text=f"Drawing Time: {self.time} seconds")

        # updates if time is not negative
        if self.time != 0:
            self.root.after(1000, self.update_timer)

        else:
            # completes the drawing time
            self.drawing_time = False

            # closes the drawing window
            self.root.destroy()

    # draws a pixel on the screen on mouse pressed position
    def draw_pixel(self, event):
        # checks whether drawing time is up
        if not self.drawing_time:
            # start the timer only when the first pixel is drawn
            self.drawing_time = True
            self.update_timer()

        # calculates grid position to place pixels
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        # check if the coordinates are within the valid range
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # draw a 2x2 pixel by adjusting coordinates
            self.canvas.create_rectangle(
                col * self.cell_size, row * self.cell_size,
                (col + 2) * self.cell_size, (row + 2) * self.cell_size,
                fill="black"
            )

            # mark cells in the grid
            for r in range(row, min(row + 2, self.rows)):
                for c in range(col, min(col + 2, self.cols)):
                    self.grid[r][c] = 1

    # run the main loop
    def run(self) -> np.array:
        self.root.mainloop()
        return self.grid


# the number game section 
class NumberGame:
    def __init__(self, time_limit, player_amount, rounds) -> None:
        # time limit on drawing
        self.time_limit: int = time_limit

        # rounds in a game
        self.rounds: int = rounds
        self.elapsed_rounds: int = 0

        # player scoring and rotation
        self.players: dict = {x + 1: 0 for x in [n for n in range(player_amount)]}
        self.active_player: int = 1

    # normalise the grid data, so it is compatible with the neural network
    @staticmethod
    def normalise_grid(drawn_grid: list) -> np.array:
        normalised_grid = np.array(drawn_grid)
        normalised_grid = normalised_grid.reshape(-1)
        return normalised_grid / 1.0

    # get whether all rounds have elapsed and asks to quit the game
    def get_state(self) -> bool:
        # checks whether all rounds have elapsed
        if self.elapsed_rounds % self.rounds == 0:
            # zero check so it doesn't ask constantly for round 0
            if self.elapsed_rounds == 0:
                return False

            # ask to quit after n rounds
            if ((input("Quit: ")).upper()).startswith("Y"):
                return True
        return False

    # switch player turns and increase rounds elapsed
    def switch_player(self):
        if (self.active_player + 1) <= len(self.players):
            self.active_player += 1
        else:
            self.active_player = 1

    # scoring for players
    def scoring(self, prediction, prompt):
        # gets the index of largest number (confident prediction, correct prompt)
        prediction: int = np.argmax(prediction)
        prompt: int = np.argmax(prompt)

        # increases active player's score if correct
        if prediction == prompt:
            self.players[self.active_player] += 1

    # print the final player scores and data
    def print_score(self):
        print("============================= Scores =============================")
        for player in self.players:
            print(
                f"[Player {player}] Score: {self.players[player]}, "
                f"Accuracy: {self.players[player] / self.elapsed_rounds * 100 :.2f}%")

    # updates round count after every player has completed a drawing
    def round_update(self):
        if self.active_player == len(self.players):
            self.elapsed_rounds += 1

    # main prediction and drawing instance loop
    def run(self):
        prompt = np.random.randint(10)

        # one hot encoding the prompt value for the neural network
        prompt_vector = [0] * 10
        prompt_vector[prompt] = 1

        # create an instance of the DrawingGrid class
        drawing_grid = DrawingGrid(rows=28, cols=28, cell_size=20, prompt=prompt, time_limit=self.time_limit,
                                   player=self.active_player)

        # start the application
        drawn_grid = drawing_grid.run()
        input_matrix = self.normalise_grid(drawn_grid)

        # send input data to neural network to get prediction
        return input_matrix, prompt_vector


# gets the user input and converts to numeric
def get_numeric_input(text: str, default, min_value, max_value, type_int: bool = True):
    value = ""
    while isinstance(value, str):
        try:
            user_input = input(f"{text} (range {min_value} - {max_value}): ")

            # if no value is input return the default
            if not user_input:
                return default

            # choose return type
            if type_int:
                value = int(user_input)
            else:
                value = float(user_input)

            # bound check to make sure the value is within range
            if max_value < value or value < min_value:
                raise ValueError

            return value

        # exception handling for value errors
        except ValueError:
            print("Invalid Entry")
            value = ""


# main loop
def main():
    # print title
    print("========================= Neural Network =========================")

    # create instance of the neural network
    ff_neural_network = FeedForwardModel()

    # load previous trained parameters
    imported: bool = False
    if (input("Import Network Parameters: ").upper()).startswith("Y"):
        ff_neural_network.import_values(input('File Name: '))
        imported = True

    # else; train the neural network
    else:
        ff_neural_network.train()

    # asks if they want to use the game version
    game: bool = False
    if ((input("Play Game Version: ")).upper()).startswith("Y"):
        game = True
        automatic = True

    else:
        # gets whether to train the neural network automatically or test via user
        automatic = False
        if ((input("Automatic Training: ")).upper()).startswith("Y"):
            automatic = True

    # use the neural network to predict an input
    print("=========================== Prediction ===========================")
    images, labels = ff_neural_network.get_dataset(False)  # get testing dataset
    correct: int = 0  # for calculating accuracy
    predictions: int = 0

    # test the neural network
    if automatic:
        for predictions in range(10000):  # tests every image in dataset
            if ff_neural_network.predict(images[predictions], labels[predictions], False):
                correct += 1

    # manually test neural network            
    else:
        while True:
            index = np.random.randint(1, 10000)
            ff_neural_network.predict(images[index], labels[index], True)
            predictions += 1

            # tests until closed by user
            if ((input("Stop Testing: ")).upper()).startswith("Y"):  # fix bug with crash on "" entry
                break

    # print final accuracy
    print(f"Estimated Accuracy: {correct / (predictions + 1) * 100:.2f}%, Number of Tests: {predictions + 1}")

    # the number prediction game
    if game:
        print("=================== Neural Network Number Game ===================")
        # get game variables
        player_amount = get_numeric_input("Player Amount", 1, 1, 20)
        time_limit = get_numeric_input("Timer Length", 5, 2, 60)
        rounds = get_numeric_input("Rounds", 10, 1, 1000)

        # display settings
        print("="*66)
        print(f"Player Amount: {player_amount}, Timer Length: {time_limit}, Rounds: {rounds}")
        print("="*66)

        # create game instance
        number_game = NumberGame(time_limit, player_amount, rounds)

        # main game loop
        while True:
            # gets drawing
            drawing, prompt = number_game.run()

            # feeds into neural network
            ff_neural_network.predict(drawing, prompt, True)

            # updates players score
            number_game.scoring(ff_neural_network.prediction, prompt)

            # updates the current round counter
            number_game.round_update()

            # switches player
            number_game.switch_player()

            # checks whether players quit
            if number_game.get_state():
                break

        # print end game info
        number_game.print_score()

    print("="*66)
    if not imported:  # checks if file already exists, due to import
        if ((input("Export Network Parameters: ")).upper()).startswith("Y"):
            ff_neural_network.export_values(input("File Name: "))


if __name__ == "__main__":
    main()
