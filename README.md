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
на 0,6 пункта выше офлайн-режима.

### Установка

```bash
pip install "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM"
pip install git+https://github.com/Sprytin/GigaAM-v3-Streaming
```

Нужны Python 3.10+ и GPU с CUDA: на процессоре потоковый режим не успевает
за речью.

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
записях, с той же нормализацией (gigastt 2.22.0, `e2e_rnnt`, 3 потока CPU):

| | gigastt | GigaAM-v3 Streaming |
|---|---:|---:|
| WER офлайн / потоково | 7,05% / 12,41% | 7,01% / 7,57% |
| CER офлайн / потоково | 3,76% / 8,75% | 3,83% / 3,97% |
| Первый текст после начала слова (p50 / p95) | 0,78 / 1,62 с | 0,42 / 0,55 с |
| Готовая фраза после конца последнего слова (p50 / p95) | 1,22 / 1,95 с | 1,60 / 2,09 с |
| Время на секунду звука, потоково | 0,67 с CPU | 0,52 с GPU |

Офлайн точность одинаковая. В потоке gigastt чаще режет фразу на паузах и
теряет или повторяет слова на стыках окон; зато ему хватает CPU, а финал
приходит раньше. Берите gigastt, если нужен готовый сервер без GPU; эту
библиотеку, если важна точность.

### Ограничения

* Нужен GPU с CUDA: в 8-10 раз больше вычислений, чем офлайн.
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

### Install

```bash
pip install "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM"
pip install git+https://github.com/Sprytin/GigaAM-v3-Streaming
```

Requires Python 3.10+ and a CUDA GPU: on CPU, streaming falls behind speech.

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
normalization (gigastt 2.22.0, `e2e_rnnt`, 3 CPU threads):

| | gigastt | GigaAM-v3 Streaming |
|---|---:|---:|
| WER offline / streaming | 7.05% / 12.41% | 7.01% / 7.57% |
| CER offline / streaming | 3.76% / 8.75% | 3.83% / 3.97% |
| First text after a word starts (p50 / p95) | 0.78 / 1.62 s | 0.42 / 0.55 s |
| Finished phrase after the last word ends (p50 / p95) | 1.22 / 1.95 s | 1.60 / 2.09 s |
| Time per second of audio, streaming | 0.67 s CPU | 0.52 s GPU |

Offline accuracy is the same. Streaming, gigastt splits phrases at pauses
more often and loses or repeats words where its windows join; in return it
runs on a CPU and its finals come sooner. Pick gigastt for a ready server
without a GPU; pick this library for accuracy.

### Limits

* Needs a CUDA GPU: 8-10 times the offline compute.
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
