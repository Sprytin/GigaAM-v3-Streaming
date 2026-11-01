# GigaAM-v3 Streaming

[![test](https://github.com/Sprytin/GigaAM-v3-Streaming/actions/workflows/test.yml/badge.svg)](https://github.com/Sprytin/GigaAM-v3-Streaming/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Streaming Russian speech recognition on the stock GigaAM v3 model: live
partials, finals with word times, the same weights and almost offline
accuracy.

[Русский](#русский) · [English](#english)

## Русский

[GigaAM](https://github.com/salute-developers/GigaAM) от Сбера входит в число
самых точных открытых моделей распознавания русской речи, но распознаёт
только готовую запись. Для живого голоса (ассистент, субтитры, диктовка,
звонки) нужен текст, пока человек говорит.

GigaAM-v3 Streaming делает стоковую GigaAM v3 потоковой: подаёте звук
кусками по мере поступления, черновой текст появляется примерно через 0,4 с
после начала слова, готовая фраза с таймингами слов примерно через 1,6 с
после её конца. Те же веса, без дообучения и без отдельной VAD-модели; WER
на 0,6 пункта выше офлайн-режима. Работает на GPU или на процессоре: три
режима с честно измеренными компромиссами, см. [Выбор режима](#выбор-режима).

### Установка

```bash
pip install "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM"
pip install git+https://github.com/Sprytin/GigaAM-v3-Streaming
# режимы CPU: то же с ONNX Runtime
pip install "gigaam-v3-streaming[cpu] @ git+https://github.com/Sprytin/GigaAM-v3-Streaming"
```

Нужен Python 3.10+. Режиму GPU нужна CUDA, режимам CPU хватает процессора.

### Быстрый старт

```python
from gigaam_v3_streaming import Final, StreamingRecognizer

recognizer = StreamingRecognizer.from_pretrained("v3_e2e_rnnt")  # скачает веса
for chunk in microphone():  # 16 кГц моно: float32, int16 или байты s16le
    for event in recognizer.feed(chunk):
        if isinstance(event, Final):
            print(event.text, [(w.text, w.start, w.end) for w in event.words])
        else:
            print("...", event.text)  # черновик, может уточниться
for event in recognizer.finish():  # конец потока: дораспознать хвост
    print(event.text)
```

Распознать файл и WebSocket-сервер: [`examples/`](examples/), API:
[docs/usage.md](docs/usage.md).

### Выбор режима

Одна и та же модель на тех же 898 записях, три режима. Все числа взяты из
[`benchmarks/results/`](benchmarks/results/) скриптом
`benchmarks/modes_table.py`.

<!-- modes:ru -->
|  | GPU | CPU | CPU + черновики |
| --- | ---: | ---: | ---: |
| Нужно | GPU с CUDA | 3 потока CPU | 3 потока CPU |
| Где мерили | RTX 4070 SUPER | Ryzen 5 7500F | Ryzen 5 7500F |
| WER / CER | 7,57% / 3,97% | 7,52% / 4,05% | 7,50% / 4,05% |
| Первый текст после начала слова (p50 / p95) | 0,42 / 0,55 с | 1,53 / 1,82 с | 0,65 / 0,85 с |
| Готовая фраза после конца последнего слова (p50 / p95) | 1,60 / 2,09 с | 1,98 / 2,48 с | 2,30 / 2,71 с |
| Черновой текст, пока слово звучит | да | нет | да |
| Счёт на секунду звука | 0,52 с GPU | 0,50 с CPU | 0,98 с CPU |
| Память | 469 МБ GPU | 928 МБ ОЗУ | 952 МБ ОЗУ |
| Запуск нового процесса, файлы в кэше | — | 2,1 с | 2,1 с |
| Загрузка | `load_engine("gpu")` | `load_engine("cpu")` | `load_engine("cpu-preview")` |
<!-- /modes:ru -->

- **GPU**: самый быстрый текст, черновик через 0,4 с после начала слова.
  Нужен GPU с CUDA; одна RTX 4070 SUPER тянет 2-3 потока в реальном времени.
- **CPU**: без видеокарты. Кодировщик работает в ONNX Runtime с весами INT8,
  точность та же, что на GPU. Цена в задержке: текст приходит только
  готовыми кусками с шагом 0,96 с, поэтому первые слова видны примерно через
  1,5 с, а готовая фраза примерно на 0,4 с позже, чем на GPU. Подходит для
  субтитров, расшифровки звонков и диктовки, хуже для голосового ассистента.
- **CPU + черновики**: добавляет черновой текст через 0,65 с после начала
  слова, но требует вдвое больше CPU (около одного ядра на поток), а готовая
  фраза приходит ещё примерно на 0,3 с позже.

Счёт: время на секунду звука; время CPU суммировано по всем 3 потокам ONNX
Runtime, так что машина с N ядрами вытянет не больше примерно N ÷ это число
потоков (параллельные потоки не мерили). Память CPU: пик всего процесса;
320 МБ весов кодировщика отображаются из кэша и делятся между процессами.
Первый запуск в режиме CPU один раз собирает INT8-кодировщик из весов
(около 40 с и 5 ГБ памяти) и кладёт его в кэш; «запуск» в таблице — новый
процесс с готовым кэшем до первого распознанного окна. Подробности:
[docs/benchmarks.md](docs/benchmarks.md#cpu-modes).

```python
from gigaam_v3_streaming import load_engine

engine = load_engine("cpu")  # "gpu", "cpu" или "cpu-preview"; "auto": gpu, если есть CUDA
recognizer = engine.recognizer()  # по одному на поток звука; движок общий
```

### Результаты

898 публичных русских записей (1,54 ч: Golos crowd, Golos farfield,
FLEURS), модель `v3_e2e_rnnt`, RTX 4070 SUPER, звук подаётся в темпе
реального времени.

| | GigaAM v3, офлайн | GigaAM-v3 Streaming |
|---|---:|---:|
| WER / CER | 7,01% / 3,83% | 7,57% / 3,97% |
| Первый текст после начала слова¹ (p50 / p95) | после конца записи | 0,42 / 0,55 с |
| Готовая фраза после конца последнего слова² (p50 / p95) | после конца записи | 1,60 / 2,09 с |
| Время GPU на секунду звука³ | 0,05 с | 0,52 с |
| Видеопамять, пик на один поток | 470 МБ | 469 МБ |

¹ Начало слова берётся из офлайн-GigaAM (`word_timestamps=True`).\
² Включая паузу 600 мс, после которой фраза закрывается.\
³ GPU в этом прогоне делили с другой нагрузкой; на свободной GPU 0,03 с
против 0,27 с.

Потоки не батчатся, поэтому одна RTX 4070 SUPER тянет в реальном времени
2-3 потока. Разбивка по наборам, методика и как повторить:
[docs/benchmarks.md](docs/benchmarks.md); числа по каждой записи (без
текстов): [`benchmarks/results/`](benchmarks/results/).

### Сравнение с gigastt

[gigastt](https://github.com/ekhodzitsky/gigastt) (MIT, Rust) тоже
распознаёт GigaAM v3 потоково: готовый сервер на ONNX INT8, по умолчанию на
CPU, с WebSocket, HTTP и SDK. Он перекодирует короткие окна (до 2,5 с) с
новым состоянием декодера в каждом. Здесь окна длиннее (8 с назад, 0,8 с
вперёд), а состояние декодера переносится между окнами. На тех же 898
записях, с той же нормализацией (gigastt 2.22.0, `e2e_rnnt`, 3 потока того
же CPU); справа три режима этой библиотеки:

| | gigastt | GPU | CPU | CPU + черновики |
|---|---:|---:|---:|---:|
| WER офлайн / потоково | 7,05% / 12,41% | 7,01% / 7,57% | 7,01% / 7,52% | 7,01% / 7,50% |
| CER офлайн / потоково | 3,76% / 8,75% | 3,83% / 3,97% | 3,80% / 4,05% | 3,80% / 4,05% |
| Первый текст после начала слова (p50 / p95) | 0,78 / 1,62 с | 0,42 / 0,55 с | 1,53 / 1,82 с | 0,65 / 0,85 с |
| Готовая фраза после конца последнего слова (p50 / p95) | 1,22 / 1,95 с | 1,60 / 2,09 с | 1,98 / 2,48 с | 2,30 / 2,71 с |
| Время на секунду звука, потоково | 0,67 с CPU | 0,52 с GPU | 0,50 с CPU | 0,98 с CPU |

Офлайн точность одинаковая. В потоке gigastt чаще режет фразу на паузах и
теряет или повторяет слова на стыках окон; зато финал у него приходит
раньше. На том же процессоре режим CPU здесь точнее (WER 7,52 против 12,41)
при меньшем счёте, но первые слова показывает позже (1,5 с против 0,8); с
черновиками раньше (0,65 с), за полтора раза больший счёт. Берите gigastt,
если нужен готовый сервер или самая быстрая готовая фраза; эту библиотеку,
если важна точность, на GPU или на CPU.

### Ограничения

* Поток стоит в 8-10 раз больше вычислений, чем офлайн: на GPU, или на CPU
  ценой задержки (первые слова через 1,5 с, черновики через 0,65 с за вдвое
  больший счёт).
* Режимы CPU измерены на одном процессоре x86-64; ARM не проверялся.
* Нет батчинга между потоками.
* Только жадное декодирование, без beam search и языковой модели.
* Только 16 кГц моно.
* Поток измерен на `v3_e2e_rnnt` (с пунктуацией); `v3_rnnt` устроена так же.
  Старые RNN-T чекпоинты GigaAM загружаются, но не проверялись; CTC не
  поддерживается.

### Документация

* [Использование и API](docs/usage.md)
* [Как это работает](docs/how-it-works.md)
* [Бенчмарки](docs/benchmarks.md)
* [Изменения](CHANGELOG.md)

### Лицензия

MIT. GigaAM и её веса под MIT от GigaChat Team; лицензии зависимостей и
данных: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## English

[GigaAM](https://github.com/salute-developers/GigaAM) by Sber is one of the
most accurate open Russian speech recognition models, but it only
recognizes finished recordings. Live voice (assistants, captions, dictation,
calls) needs the text while the person is still speaking.

GigaAM-v3 Streaming turns the stock GigaAM v3 into a streaming recognizer:
feed audio as it arrives, get draft text about 0.4 s after a word starts and
the finished phrase with word times about 1.6 s after it ends. Same weights,
no retraining, no separate VAD model; WER 0.6 points above offline decoding.
It runs on a GPU or on a CPU: three modes with honestly measured trade-offs,
see [Choose your mode](#choose-your-mode).

### Install

```bash
pip install "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM"
pip install git+https://github.com/Sprytin/GigaAM-v3-Streaming
# CPU modes: the same with ONNX Runtime
pip install "gigaam-v3-streaming[cpu] @ git+https://github.com/Sprytin/GigaAM-v3-Streaming"
```

Requires Python 3.10+. The GPU mode needs CUDA; the CPU modes need only a
CPU.

### Quick start

```python
from gigaam_v3_streaming import Final, StreamingRecognizer

recognizer = StreamingRecognizer.from_pretrained("v3_e2e_rnnt")  # downloads weights
for chunk in microphone():  # 16 kHz mono: float32, int16 or s16le bytes
    for event in recognizer.feed(chunk):
        if isinstance(event, Final):
            print(event.text, [(w.text, w.start, w.end) for w in event.words])
        else:
            print("...", event.text)  # draft, may still change
for event in recognizer.finish():  # end of stream: decode the tail
    print(event.text)
```

File transcription and a WebSocket server: [`examples/`](examples/); API:
[docs/usage.md](docs/usage.md).

### Choose your mode

The same model on the same 898 clips in three modes. Every number comes from
[`benchmarks/results/`](benchmarks/results/) through
`benchmarks/modes_table.py`.

<!-- modes:en -->
|  | GPU | CPU | CPU + preview |
| --- | ---: | ---: | ---: |
| Needs | CUDA GPU | 3 CPU threads | 3 CPU threads |
| Measured on | RTX 4070 SUPER | Ryzen 5 7500F | Ryzen 5 7500F |
| WER / CER | 7.57% / 3.97% | 7.52% / 4.05% | 7.50% / 4.05% |
| First text after a word starts (p50 / p95) | 0.42 / 0.55 s | 1.53 / 1.82 s | 0.65 / 0.85 s |
| Finished phrase after the last word ends (p50 / p95) | 1.60 / 2.09 s | 1.98 / 2.48 s | 2.30 / 2.71 s |
| Draft text while a word is spoken | yes | no | yes |
| Compute per second of audio | 0.52 s GPU | 0.50 s CPU | 0.98 s CPU |
| Memory | 469 MB GPU | 928 MB RAM | 952 MB RAM |
| Start, fresh process, files cached | — | 2.1 s | 2.1 s |
| Load it | `load_engine("gpu")` | `load_engine("cpu")` | `load_engine("cpu-preview")` |
<!-- /modes:en -->

- **GPU** gives the fastest text: drafts 0.4 s after a word starts. It needs
  a CUDA GPU; one RTX 4070 SUPER runs 2-3 streams in real time.
- **CPU** needs no GPU. The encoder runs in ONNX Runtime with INT8 weights
  and accuracy stays where it is on the GPU. The cost is delay: text comes
  only as finished pieces in 0.96 s steps, so the first words show about
  1.5 s after they are spoken, and a finished phrase about 0.4 s later than
  on the GPU. Good for captions, call transcripts and dictation, weaker for a
  voice assistant.
- **CPU + preview** adds draft text 0.65 s after a word starts, but takes
  twice the CPU (about one core per stream), and the finished phrase comes
  about 0.3 s later still.

Compute is time spent per second of audio. GPU time was measured on a card
shared with another workload; on an idle card it is 0.27 s
(`v3_e2e_rnnt_memory.json`). CPU time is summed over all 3 ONNX Runtime
threads, so a machine with N cores has room for at most about N ÷ that
number streams; parallel streams were not measured. GPU memory is the CUDA
peak for one stream; CPU memory is the peak resident memory of the whole
process, and the encoder's 320 MB of weights in it are memory-mapped from
the cache, so processes on one machine share them. The first CPU start
builds the INT8 encoder from the checkpoint once (about 40 s and 5 GB of
memory) and caches it; start time is a fresh process with the cache ready,
up to its first recognized window. Details:
[docs/benchmarks.md](docs/benchmarks.md#cpu-modes).

```python
from gigaam_v3_streaming import load_engine

engine = load_engine("cpu")  # "gpu", "cpu" or "cpu-preview"; "auto": gpu if CUDA is there
recognizer = engine.recognizer()  # one per audio stream; streams share the engine
```

### Results

898 public Russian clips (1.54 h: Golos crowd, Golos farfield, FLEURS),
model `v3_e2e_rnnt`, RTX 4070 SUPER, audio fed at real-time pace.

| | GigaAM v3, offline | GigaAM-v3 Streaming |
|---|---:|---:|
| WER / CER | 7.01% / 3.83% | 7.57% / 3.97% |
| First text after a word starts¹ (p50 / p95) | after the recording ends | 0.42 / 0.55 s |
| Finished phrase after the last word ends² (p50 / p95) | after the recording ends | 1.60 / 2.09 s |
| GPU time per second of audio³ | 0.05 s | 0.52 s |
| GPU memory, peak for one stream | 470 MB | 469 MB |

¹ Word starts come from offline GigaAM (`word_timestamps=True`).\
² Including the 600 ms pause after which the phrase closes.\
³ The GPU was shared with another workload in this run; on an idle GPU
0.03 s against 0.27 s.

Streams are not batched, so one RTX 4070 SUPER runs 2-3 streams in real
time. Per-set numbers, the method and how to reproduce them:
[docs/benchmarks.md](docs/benchmarks.md); per-clip numbers (no
transcripts): [`benchmarks/results/`](benchmarks/results/).

### Comparison with gigastt

[gigastt](https://github.com/ekhodzitsky/gigastt) (MIT, Rust) also streams
GigaAM v3: a ready server on ONNX INT8, CPU by default, with WebSocket, HTTP
and SDKs. It re-encodes short windows (up to 2.5 s) with a fresh decoder
state in each. Here windows are longer (8 s back, 0.8 s ahead) and decoder
state carries across windows. On the same 898 clips with the same
normalization (gigastt 2.22.0, `e2e_rnnt`, 3 threads of the same CPU); the
last three columns are this library's modes:

| | gigastt | GPU | CPU | CPU + preview |
|---|---:|---:|---:|---:|
| WER offline / streaming | 7.05% / 12.41% | 7.01% / 7.57% | 7.01% / 7.52% | 7.01% / 7.50% |
| CER offline / streaming | 3.76% / 8.75% | 3.83% / 3.97% | 3.80% / 4.05% | 3.80% / 4.05% |
| First text after a word starts (p50 / p95) | 0.78 / 1.62 s | 0.42 / 0.55 s | 1.53 / 1.82 s | 0.65 / 0.85 s |
| Finished phrase after the last word ends (p50 / p95) | 1.22 / 1.95 s | 1.60 / 2.09 s | 1.98 / 2.48 s | 2.30 / 2.71 s |
| Time per second of audio, streaming | 0.67 s CPU | 0.52 s GPU | 0.50 s CPU | 0.98 s CPU |

Offline accuracy is the same. Streaming, gigastt splits phrases at pauses
more often and loses or repeats words where its windows join; in return its
finals come sooner. On the same CPU, the CPU mode here is more accurate
(WER 7.52 against 12.41) for less compute but shows first words later
(1.5 s against 0.8 s); with preview it shows them sooner (0.65 s) for about
1.5 times the compute. Pick gigastt for a ready server or the fastest
finished phrase; pick this library for accuracy, on a GPU or a CPU.

### Limits

* Streaming costs 8-10 times the offline compute: on a GPU, or on a CPU at
  the cost of delay (first words at 1.5 s, or drafts at 0.65 s for twice the
  compute).
* The CPU modes were measured on one x86-64 CPU; ARM is untested.
* No batching across streams.
* Greedy decoding only; no beam search or language model.
* 16 kHz mono input only.
* Streaming measured on `v3_e2e_rnnt` (with punctuation); `v3_rnnt` has the
  same architecture. Older GigaAM RNN-T checkpoints load but are untested;
  CTC is not supported.

### Documentation

* [Usage and API](docs/usage.md)
* [How it works](docs/how-it-works.md)
* [Benchmarks](docs/benchmarks.md)
* [Changelog](CHANGELOG.md)

### License

MIT. GigaAM and its weights are MIT, GigaChat Team; dependency and data
licenses: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
