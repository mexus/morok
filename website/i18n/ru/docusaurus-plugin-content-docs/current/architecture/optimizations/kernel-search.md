---
sidebar_label: Поиск по ядрам
---

# Поиск оптимизации ядер

После алгебраического упрощения каждому ядру нужны *решения по планированию*: как тайлить циклы, где параллелизовать, использовать ли tensor cores. Morok предлагает две стратегии: быстрые эвристики и тщательный beam search.

Выполняется на Стадии 7 [пайплайна кодогенерации](../codegen/overview.md).

Исходники Tinygrad: `tinygrad/codegen/opt/`. Исходники Morok: `schedule/src/optimizer/`.

---

## Пространство действий

Оптимизационные преобразования модифицируют структуру циклов, меняя типы осей. Каждое действие изменяет один диапазон:

| Действие | Эффект | Целевое оборудование |
|----------|--------|---------------------|
| UPCAST(axis, amount) | Векторизация размерности (SIMD) | Все |
| UNROLL(axis, amount) | Развёртка размерности цикла | Все |
| LOCAL(axis, amount) | Использование GPU shared memory | GPU (LDS) / CPU (L1) |
| GROUP(axis, amount) | Двухстадийная редукция | Все |
| GROUPTOP(axis, amount) | Grouped reduction для tensor cores | GPU |
| THREAD(axis, amount) | Параллелизация на основе потоков CPU | CPU |
| SWAP(axis1, axis2) | Перестановка глобальных размерностей | Все |
| PADTO(axis, amount) | Паддинг для выравнивания | Все |
| NOLOCALS | Отключение локальной памяти | Все (ограничение) |
| TC | Включение использования tensor cores | GPU NVIDIA |

Суммарное пространство действий — ~162 базовых действия (зависит от структуры ядра и доступного параллелизма).

---

## Эвристики (по умолчанию)

Эвристический оптимизатор применяет оптимизации в фиксированном порядке (упрощённый псевдокод):

```rust
// Pseudocode — simplified from optimizer/heuristics.rs
fn hand_coded_optimizations(scheduler: &mut Scheduler) {
    // 1. Tensor cores (if matmul pattern detected)
    if let Some(tc) = detect_tensor_core_pattern(scheduler) {
        apply_tensor_core(scheduler, tc);
        return;  // TC handles everything
    }

    // 2. Grouped reductions (two-stage for large reductions)
    apply_grouped_reduction_if_needed(scheduler);

    // 3. Vectorization (UPCAST output dimensions)
    apply_upcast(scheduler, 4);

    // 4. GPU local memory (workgroup dimensions)
    apply_local_dims(scheduler);

    // 5. CPU threading
    apply_threading(scheduler);
}
```

**Плюсы**: Быстро (~50ms на ядро), предсказуемо, не требует аппаратных замеров.

**Минусы**: Может упустить возможности оптимизации, фиксированные эвристики не адаптируются к нагрузке.

---

## Beam search (опционально)

Для продакшн-нагрузок beam search находит лучшие расписания, компилируя и замеряя кандидатов (упрощённый псевдокод):

```rust
// Pseudocode — simplified from optimizer/beam.rs
// Actual API: beam_search_cached(scheduler, config, compile_and_time) -> Result<BeamResult>
fn beam_search(scheduler: Scheduler, config: BeamConfig) -> Scheduler {
    let mut beam = vec![scheduler];
    let deadline = Instant::now() + config.time_limit;

    while Instant::now() < deadline {
        let mut candidates = vec![];

        for state in &beam {
            for action in generate_actions(state) {
                if let Ok(next) = state.apply(action) {
                    candidates.push(next);
                }
            }
        }

        // Compile and time each candidate
        let timed: Vec<_> = candidates.par_iter()
            .map(|c| (c, measure_kernel_time(c)))
            .collect();

        // Keep top K by execution time
        beam = timed.into_iter()
            .sorted_by_key(|(_, time)| *time)
            .take(config.beam_width)
            .map(|(c, _)| c)
            .collect();
    }

    beam.into_iter().next().unwrap()
}
```

**Плюсы**: Находит близкие к оптимальным расписания, адаптируется к железу.

**Минусы**: Минуты на ядро (но результаты кэшируются по хэшу AST).

---

## Конфигурация

```bash
# Disable optimization (debugging)
MOROK_NOOPT=1 cargo run

# Enable beam search with width 8
MOROK_BEAM=8 cargo run
```

Или программно:

```rust
let config = PrepareConfig::builder()
    .strategy(OptStrategy::Beam { width: 8 })
    .build();

tensor.realize_with(config)?;
```

---

## Сравнение: как оптимизируют другие компиляторы

| Аспект | XLA | TVM/Ansor | Triton | **Morok** |
|--------|-----|-----------|--------|-----------|
| **Философия** | Фиксированные эвристики | Поиск | Управление программистом | На основе паттернов |
| **Фьюзинг** | Консервативные правила | Tile-and-fuse | Block-level | Перезапись графа |
| **Автотюнинг** | Нет | Эволюционный + cost model | Grid search | Beam search |
| **Стоимость тюнинга** | 0 | Часы | Минуты | Минуты (кэшируется) |
| **Гибкость** | Низкая | Высокая | Средняя | Высокая |
| **Прозрачность** | Низкая (C++-проходы) | Средняя (Python) | Средняя (DSL) | Высокая (декларативные паттерны) |

**XLA** использует фиксированные эвристики для решений по фьюзингу. Безопасно и предсказуемо, но оставляет производительность на столе. Правила фьюзинга захардкожены в C++.

**TVM/Ansor** разделяет *что* вычислять и *как* вычислять. Ansor использует эволюционный поиск с обучаемой cost model. Лучшая в классе производительность, но тюнинг занимает часы на модель.

**Triton** предоставляет Python-подобный DSL для блочных алгоритмов. Хороший баланс контроля и автоматизации, но требует экспертизы в GPU-программировании.

**Morok** выражает оптимизации как компонуемые паттерны. Beam search добавляет автотюнинг при необходимости, с кэшированием результатов по хэшу AST для повторного использования.
