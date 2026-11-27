from sklearn.dummy import DummyClassifier


def make_dummy_model(*, random_state=0):
    return DummyClassifier(
        strategy="stratified",
        random_state=random_state,
    )