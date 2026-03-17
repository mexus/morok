---
sidebar_label: ONNX-инференс
---

# Инференс ONNX-моделей

ONNX-импортёр Morok — рекомендуемый способ инференса моделей. Он загружает стандартные `.onnx`-файлы, раскладывает операторы на ленивые тензорные операции Morok и компилирует их через полный пайплайн оптимизаций — без C++ рантайма.

**Текущий статус:**

| Возможность | Статус |
|-------------|--------|
| Прямой инференс | Поддерживается |
| 162 / 200 операторов ONNX | [Таблица паритета](https://github.com/patsak/morok/blob/main/onnx/PARITY.md) |
| CNN-архитектуры (ResNet, DenseNet, VGG, ...) | Проверено 9 моделей |
| Расширения Microsoft (Attention, RotaryEmbedding) | Поддерживается |
| Динамический размер батча | Планируется в следующем релизе |
| Обучение / обратный проход | Не поддерживается |

**Сравнение с другими фреймворками**

Среди чистых Rust-фреймворков у Morok самое широкое покрытие операторов ONNX — 162 оператора, 1361 пройденный conformance-тест на двух бэкендах (Clang + LLVM). У `candle` и `burn` операторов меньше, а тестовых наборов сопоставимого масштаба нет. Если же нужна максимальная совместимость с продакшн-моделями ONNX — используйте `ort`, Rust-обёртку вокруг C++ ONNX Runtime, которая покрывает полную спецификацию.

---

## Быстрый старт

Добавьте `morok-onnx` и `morok-tensor` в `Cargo.toml`:

```toml
[dependencies]
morok-onnx = { git = "https://github.com/patsak/morok" }
morok-tensor = { git = "https://github.com/patsak/morok" }
```

### Простой вариант: модели со встроенными весами

Для моделей, у которых все входы уже вшиты в файл (без рантайм-входов):

```rust
use morok_onnx::OnnxImporter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut importer = OnnxImporter::new();
    let outputs = importer.import_path("model.onnx")?;

    // Each output is a lazy Tensor — realize to get data
    for (name, tensor) in &outputs {
        let result = tensor.realize()?;
        println!("{name}: {:?}", result.to_ndarray::<f32>()?);
    }
    Ok(())
}
```

### Двухфазный вариант: модели с рантайм-входами

Большинству моделей нужны данные на этапе выполнения (изображения, токены, аудио). Двухфазный API разделяет подготовку графа и выполнение:

```rust
use morok_onnx::OnnxImporter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let importer = OnnxImporter::new();

    // Phase 1: Parse structure (no execution)
    let graph = importer.prepare(model_proto)?;

    // Inspect what the model needs
    for (name, spec) in &graph.inputs {
        println!("{name}: shape={:?}, dtype={:?}", spec.shape, spec.dtype);
    }

    // Phase 2: Build lazy computation graph
    let (inputs, outputs) = importer.trace(&graph)?;

    // Execute
    let result = outputs["output"].realize()?;
    Ok(())
}
```

`prepare()` и `trace()` разделены, потому что структура графа статична — парсим один раз, а `trace()` можно вызывать многократно с разными привязками размерностей или входными данными.

---

## Архитектура

### Двухфазный дизайн

Импортёр обрабатывает ONNX-модели в два этапа:

**Фаза 1 — `prepare()`:** Извлекает топологию графа, ничего не выполняя. Парсит protobuf, отделяет инициализаторы (веса) от рантайм-входов, запоминает версии opset, предварительно парсит подграфы control flow. Возвращает `OnnxGraph` — лёгкую структуру, которую можно изучить до запуска вычислений.

**Фаза 2 — `trace()`:** Обходит граф в топологическом порядке, диспатчит каждый ONNX-узел в соответствующую реализацию Tensor. На этом этапе строится ленивый граф вычислений (DAG) — никаких реальных вычислений ещё нет. Результат — набор хэндлов `Tensor`, которые при вызове `realize()` компилируются и выполняются через полный пайплайн.

```text
model.onnx → prepare() → OnnxGraph → trace() → lazy Tensors → realize() → results
                 │                        │
                 │ structure only         │ builds computation DAG
                 │ (no execution)         │ (no execution)
                 ▼                        ▼
          Inspect inputs/outputs    Pass to optimizer/codegen
```

Такое разделение даёт несколько возможностей:
- **Проверка перед запуском:** Посмотреть формы и типы данных входов до аллокаций
- **Множественные трассировки:** Повторный `trace()` с другими привязками динамических размерностей
- **Внешние веса:** Загрузка весов отдельно (полезно для моделей с внешними файлами данных)

### Декомпозиция операторов

Каждый оператор ONNX раскладывается на операции Morok Tensor. Степень сложности разная:

**Прямые отображения** — около 60 операторов напрямую соответствуют одному методу тензора:

```rust
// In the registry:
"Add" => x.try_add(y)?
"Relu" => x.relu()?
"Sigmoid" => x.sigmoid()?
"Equal" => x.try_eq(y)?
```

**Паттерны-билдеры** — сложные операторы с множеством необязательных параметров используют fluent API:

```rust
// Conv with optional bias, padding, dilation, groups
x.conv()
    .weight(w)
    .maybe_bias(bias)
    .auto_pad(AutoPad::SameLower)
    .group(32)
    .maybe_dilations(Some(&[2, 2]))
    .call()?
```

**Многошаговые декомпозиции** — операторы вроде BatchNormalization, Attention и Mod требуют промежуточных вычислений. Например, целочисленный `Mod` в стиле Python раскладывается на truncation mod + поправку знака:

```rust
let trunc_mod = x.try_mod(y)?;
let signs_differ = trunc_mod.bitwise_xor(y)?.try_lt(&zero)?;
let needs_adj = mod_ne_zero.bitwise_and(&signs_differ)?;
trunc_mod.try_add(&y.where_(&needs_adj, &zero)?)?
```

### Валидация атрибутов

Хелпер `Attrs` работает по принципу pop — каждый вызов `attrs.int("axis", -1)` или `attrs.float("epsilon", 1e-5)` забирает атрибут из словаря. После обработки оператора `attrs.done()` проверяет, что словарь пуст. Оставшиеся атрибуты вызывают ошибку — так неполные реализации операторов ловятся на этапе трассировки, а не приводят к молчаливо неверным результатам.

### Версионирование opset

ONNX-модели объявляют импорты opset для каждого домена. Импортёр отслеживает их и передаёт версию каждому обработчику. Операторы переключают поведение в зависимости от версии — например, ось по умолчанию у Softmax сменилась с `1` (opset < 13) на `-1` (opset >= 13), а `ReduceSum` перенёс оси из атрибута во входной тензор в opset 13.

---

## Работа с моделями

### Динамические размерности

Входы ONNX могут содержать символические размерности вроде `"batch_size"` или `"sequence_length"`. Привяжите их на этапе трассировки:

```rust
let graph = importer.prepare(model)?;

// Bind symbolic dims to concrete values
let (inputs, outputs) = importer.trace_with_dims(
    &graph,
    &[("batch_size", 1), ("sequence_length", 512)],
)?;
```

Непривязанные динамические размерности дают понятную ошибку на этапе трассировки. Какие размерности динамические, можно узнать через `InputSpec::shape`:

```rust
for (name, spec) in &graph.inputs {
    for dim in &spec.shape {
        match dim {
            DimValue::Static(n) => print!("{n} "),
            DimValue::Dynamic(name) => print!("{name}? "),
        }
    }
}
```

### Внешние веса

Некоторые ONNX-модели хранят веса в отдельных файлах. Чтобы передать их, используйте `trace_external()`:

```rust
let (inputs, outputs) = importer.trace_external(
    &graph,
    external_weights,  // HashMap<String, Tensor>
)?;
```

### Расширения Microsoft

Импортёр поддерживает несколько contrib-операторов `com.microsoft`, которые часто встречаются в трансформерных моделях, экспортированных из ONNX Runtime:

| Расширение | Назначение |
|------------|-----------|
| `Attention` | Упакованная QKV-проекция с маскированием, past KV cache |
| `RotaryEmbedding` | Ротационные позиционные эмбеддинги (interleaved/non-interleaved) |
| `SkipLayerNormalization` | Fused residual + LayerNorm + масштабирование |
| `EmbedLayerNormalization` | Эмбеддинги токенов + позиций + сегментов → LayerNorm |

Стандартные трансформерные операторы ONNX (`Attention` из домена ai.onnx) тоже поддерживаются — с grouped query attention (GQA), каузальным маскированием, past KV cache и softcap.

---

## Control flow и ограничения

### Семантика If: обе ветки всегда выполняются

Оператор `If` в ONNX — это data-dependent control flow: условие определяет, какая ветка выполняется. Ленивые вычисления Morok принципиально несовместимы с этим: на этапе трассировки ничего не выполняется, и значение условия неизвестно.

**Решение Morok:** Трассировать *обе* ветки, а потом объединить результаты через `Tensor::where_()`:

```text
ONNX:    if condition { then_branch } else { else_branch }
Morok:   then_result.where_(&condition, &else_result)
```

Это даёт подход **«трассируй один раз — запускай многократно»** — скомпилированный граф обрабатывает любое значение условия в рантайме. Но есть жёсткое ограничение: **обе ветки должны возвращать одинаковые формы и типы данных.** Модели с shape-полиморфными ветками (then-ветка возвращает `[3, 4]`, а else-ветка — `[5, 6]`) трассировать нельзя.

На практике большинство ONNX-моделей с узлами `If` укладываются в это ограничение — условная логика в них выбирает значения, а не меняет форму данных.

### Нет Loop и Scan

Итеративный control flow (`Loop`, `Scan`) не реализован. Эти операторы требуют многократной трассировки или развёртки, что не ложится на архитектуру однократной трассировки. Модели с рекуррентными паттернами обычно работают через развёрнутые операторы (LSTM, GRU, RNN реализованы как нативные ops).

### Нет батчинга (пока)

Динамический батчинг — одновременный инференс на нескольких входах — планируется в следующем релизе. Пока что размерности батча нужно привязывать к фиксированному значению на этапе трассировки через `trace_with_dims()`.

### Нет обучения

Импортёр только для инференса. Обратного прохода, вычисления градиентов и оптимизаторов нет.

### Нереализованные категории операторов

| Категория | Примеры | Причина |
|-----------|---------|---------|
| Квантизация | DequantizeLinear, QuantizeLinear | Нужна поддержка квантизованных типов в IR |
| Операции с последовательностями | SequenceConstruct, SequenceAt | Нетензорные типы не входят в систему типов Morok |
| Случайные числа | RandomNormal, RandomUniform | Stateful RNG пока не реализован |
| Обработка сигналов | DFT, STFT, MelWeightMatrix | Низкий приоритет; узкоспециализированные задачи |
| Текст | StringNormalizer, TfIdfVectorizer | Строковые типы не поддерживаются |

Для моделей с такими операторами используйте `ort` (обёртку над ONNX Runtime) — она покрывает полную спецификацию.

---

## Отладка

### Поузловая трассировка выходов

Установите уровень логирования trace, чтобы выводить промежуточные результаты:

```bash
RUST_LOG=morok_onnx::importer=trace cargo run
```

Это вызывает `realize()` для выхода каждого узла отдельно и печатает первые 5 значений — помогает при числовой бисекции, когда модель выдаёт неверные результаты. Учтите, что это ломает фьюзинг ядер (каждый узел выполняется отдельно), так что это чисто отладочный инструмент.

### Исследование графа

Чтобы понять, что нужно модели, до трассировки используйте `OnnxGraph`:

```rust
let graph = importer.prepare(model)?;

println!("Inputs:");
for (name, spec) in &graph.inputs {
    println!("  {name}: {:?} {:?}", spec.shape, spec.dtype);
}

println!("Outputs: {:?}", graph.output_names());
println!("Nodes: {}", graph.nodes.len());
println!("Initializers: {}", graph.initializers.len());
```

---

## Итого

| Аспект | Детали |
|--------|--------|
| **Точка входа** | `OnnxImporter::new()` |
| **Простой импорт** | `importer.import_path("model.onnx")?` |
| **Двухфазный режим** | `prepare()` → `trace()` / `trace_with_dims()` |
| **Операторы** | 162 / 200 ([полная таблица паритета](https://github.com/patsak/morok/blob/main/onnx/PARITY.md)) |
| **Проверенные модели** | ResNet50, DenseNet121, VGG19, Inception, AlexNet, ShuffleNet, SqueezeNet, ZFNet |
| **Бэкенды** | Clang + LLVM (идентичные результаты) |
| **Расширения** | com.microsoft Attention, RotaryEmbedding, SkipLayerNorm, EmbedLayerNorm |
| **Ограничения** | Нет обучения, нет батчинга (пока), нет Loop/Scan, shape-полиморфный If |

**Далее:** [Практические примеры](./examples) — основы работы с тензорами, или [Пайплайн выполнения](./architecture/pipeline) — чтобы разобраться, как устроена компиляция.
