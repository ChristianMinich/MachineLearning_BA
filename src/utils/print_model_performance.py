import pickle


def load_and_print_model_performance(file_path: str):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        #print(data)

        epoch_length = len(data['accuracy']) - 1
        #print(epoch_length)
        print(f'accuracy: {data["accuracy"][epoch_length]}')
        print(f'loss: {data["accuracy"][epoch_length]}')
        print(f'val_accuracy: {data["accuracy"][epoch_length]}')
        print(f'val_loss: {data["accuracy"][epoch_length]}')


if __name__ == '__main__':
    file_path = 'C:/Users/Chris/PycharmProjects/MachineLearning_BA/histories/epochs50_BS32_TS128_digit_classifier.pkl'
    load_and_print_model_performance(file_path)
