## Keypoint detection

### Instruction
1. Установить зависимости из requirements.txt
```pip install -r requirements.txt```. Будет установлен PyTorch версии 1.8.1 
2. Скачать [shape_predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) для dlib и распаковать его
3. В файле config.py указать:
   1. `DATA_DIR` - путь до папки с данными landmarks_test
   2. `LOG_DIR` - путь до папки для логов обучения
   3. `PREDICTOR_PATH` - путь до скачанного и распакованного shape_predictor (.dat)
4. Запустить скрипт `extract_boxes.py` для извлечения боксов с лицами с помощью dlib. В результате:
   1. В папках будут созданы файлы с расширением .box
   2. Создастся файл `scores_dlib.npy`, содержащий ced скоры алгоритма dlib на датасете Menpo
5. Открыть в Jupyter тетрадку `train.ipynb` и запустить все ячейки для начала обучения
6. После обучения модели для её тестирования необходимо запустить скрипт `test.py`
7. Для визуализации результатов отрыть в Jupyter тетрадку `visualize.ipynb` и запустить ячейки. В тетрадке используется файл `scores_dlib.npy`, который был создан на 4 шаге