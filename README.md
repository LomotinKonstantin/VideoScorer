# VideoScorer
Набор скриптов для комплексной оценки и суммаризации видео.
![](https://github.com/LomotinKonstantin/VideoScorer/img/scoring_example.png)

## Подготовка
Перед началом раброты нужно установить все необходимые пакеты. Их список можно взять из файла ```env.yml```. Также (при наличии conda) можно создать из этого файла среду.
```bash
$ conda env create -f env.yml
$ conda activate va
```
## Основные скрипты
#### Загрузка тестовых видео
1. *[Опционально]* Отредактировать youtube_ds.json
2. Выполнить
```bash
$ python download_youtube_videos.py "path/to/destination/folder/"
```
#### Оценка видео
Для получения информации о параметрах можно запустить скрипт с ключом ```-h```.
Скрипт выполняет покадровую оценку видео и сохраняет отчет в формате pkl в указанную папку.
Параметры:
* ***-\-path***: путь к файлу видео
* ***-\-out_dir***: путь к папке, куда будет сохранен результат
* ***-\-gpu_mem***: количество видеопамяти в мегабайтах, которое будет выделено фреймворкам. В случае неудачи скрипт завершится с ошибкой. Чтобы отключить использование GPU, нужно передать 0.
* ***-\-sbd_threshold***: порог вероятности для разметки границ шотов. По умолчанию 0.5. Обычно менять не нужно.
#### Создание summary
Для получения информации о параметрах можно запустить скрипт с ключом ```-h```.
* ***-in***: путь к файлу видео
* ***-out***: путь к папке, куда будет сохранен результат
* ***-t***: длительность summary, в секундах
* ***-\-gpu_mem***: количество видеопамяти в мегабайтах, которое будет выделено фреймворкам. В случае неудачи скрипт завершится с ошибкой. Чтобы отключить использование GPU, нужно передать 0.
* ***-\-sbd_threshold***: порог вероятности для разметки границ шотов. По умолчанию 0.5. Обычно менять не нужно.
#### Запуск оценки на датасете [TVSum](https://github.com/yalesong/tvsum)
Перед запуском нужно загрузить и распаковать датасет. Затем в скрипте ```tvsum_eval.py``` поменять константы, задающие настройки и выполнить:
```bash
$ python tvsum_eval.py
```
Для каждого видео скрипт создает подпапку в указанном каталоге, кудо сохраняет оценку и пишет логи. Итоговый отчет по запуску сохраняется в CSV.
## Использованные реализации моделей
1. [TransNet](https://github.com/soCzech/TransNet)
2. [Shot Type Classifier](https://github.com/rsomani95/shot-type-classifier)
3. [COCO SSD MobileNet by Tensorflow](https://www.tensorflow.org/lite/models/object_detection/overview) 
4. [Cascade Face Detector by OpenCV](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)












