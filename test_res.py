from fastapi.testclient import TestClient
from app import app
from io import BytesIO
from PIL import Image

def generate_fake_image(_format: str):
    in_memory_file = BytesIO()
    image = Image.new('RGB', size=(10, 5), color=(155, 0, 0))
    image.save(in_memory_file, _format)
    in_memory_file.name = f'test_image.{_format}'
    in_memory_file.seek(0)
    return in_memory_file

client = TestClient(app)

def test_upload_file(client: TestClient):
    # test_file = 'data-ver-control/data/raw/train/n03445777/ILSVRC2012_val00002314.jpeg'
    # files = {'file': ('data-ver-control/data/raw/train/n03445777/ILSVRC2012_val00002314.jpeg', open(test_file, 'wb'))}
    # response = client.post('/app', files=files)
    # # json_data = response.json()
    # assert response.status_code == 200
    # # assert json_data == 'golf ball'
    # result = response.json()
    # assert result

    fake_image = generate_fake_image("jpeg")

    response = client.post("/predict", files={'file': fake_image})
    assert response.status_code == 200
    result = response.json()
    assert "label" in result

#------------------------------------------------------------------------------------------------------------
# import pytest
# import app
# # from app import upload_file
#
# @pytest.fixture
# def classifier():
#     return app.upload_file()
#
# def test_accuracy(classifier):
#     # Правильные данные
#     data = [
#         {'text': "n03445777", 'label': "golf ball"},
#         {'text': "n03888257", 'label': "parachute"}
#     ]
#     # Обучение классификатора
#     classifier.train(data)
#     # Тестовые данные
#     test_data = [
#         {'text': "n03445777", 'label': "golf ball"},
#         {'text': "n03888257", 'label': "parachute"},
#         {'text': "n03445777", 'label': "parachute"},  # Неправильная метка
#         {'text': "n03888257", 'label': "golf ball"},  # Неправильная метка
#     ]
#     correct_predictions = 2
#     total_predictions = len(test_data)
#     for example in test_data:
#         prediction = classifier.predict(example['text'])
#         if prediction == example['label']:
#             correct_predictions += 1
#     accuracy = correct_predictions / total_predictions
#     assert accuracy >= 0.6  # Минимальная точность должна быть не менее 60%

# def test_model_performance():
#     # load the model
#     model = load_model()
#
#     # load the test data
#     X_test, y_test = load_test_data()
#
#     # time the predictions for the test data
#     start_time = time.time()
#     y_pred = model.predict(X_test)
#     end_time = time.time()
#
#     # calculate the time per prediction
#     time_per_prediction = (end_time - start_time) / len(y_pred)
#
#     # check that the time per prediction is below a certain threshold
#     assert time_per_prediction < 0.1