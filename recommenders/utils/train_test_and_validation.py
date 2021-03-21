from datetime import datetime

from sklearn.model_selection import train_test_split


class TrainTestAndValidation:
    def __init__(self, df, test_size, valid_size: float = 0.0, random_state: int = 4):
        TEST_SIZE = test_size
        train, test = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=random_state
        )
        if valid_size:
            VALID_SIZE = valid_size
            train, validation = train_test_split(
                df,
                test_size=VALID_SIZE,
                random_state=random_state
            )
        else:
            validation = None
        print(f'{datetime.now()} - TRAINING DATA')
        print('Shape of input sequences: {}'.format(train.shape))
        print("-" * 50)
        if valid_size:
            print(f'{datetime.now()} - VALIDATION DATA')
            print('Shape of input sequences: {}'.format(validation.shape))
            print("-" * 50)
        print(f'{datetime.now()} - TESTING DATA')
        print('Shape of input sequences: {}'.format(test.shape))

        self._train, self._test, = train, test
        self._validation = validation

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def validation(self):
        return self._validation
