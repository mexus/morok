---
sidebar_label: Фаза 2 — Expander
---

# Фаза 2: Expander

**Цель**: Трансформировать оптимизационные примитивы (UNROLL/UPCAST) в явные операции.

---

## Стадия 8: Пост-оптимизационная символика

> **Стадия кратко**
>
> **Цель**: Символическое упрощение после оптимизации
> **Ключевые паттерны**: Перемещение WHERE, свёртка констант
> **Эффект**: Улучшает комбинирование загрузок и векторизацию

**Что делает**: Символическое упрощение после оптимизации, плюс перемещение WHERE.

**Зачем это нужно**: WHERE-операции — это аналог `if`-выражений. Эта стадия переносит `if`-проверки с позиции после загрузки на позицию до неё. Железо может пропустить загрузку, когда условие ложно — экономия пропускной способности памяти.

**Паттерн**: `sym + pm_move_where_on_load`

```text
// Before: WHERE guards a load
WHERE(valid, LOAD(index), alt)

// After: validity moved to INDEX
LOAD(INDEX(ptr, idx, valid=valid), alt)
```

Перенос валидности в INDEX улучшает комбинирование загрузок и векторизацию.

**Примечание**: Этот паттерн срабатывает только когда альтернативное значение равно `0`. Трансформация включает сложный анализ клауз: обнаружение дубликатов, проверки зависимостей от RANGE, верификацию data-dependent загрузок.

**Примечание**: Реализация Morok использует `gate=` вместо `valid=` (у структуры Index есть поле `gate`). Концепция идентична.

**Morok**: `pm_move_where_on_load()` в `symbolic/patterns.rs`

---

## Стадия 9: Expander

> **Стадия кратко**
>
> **Цель**: Преобразовать UNROLL/UPCAST в явные операции
> **Ключевые концепции**: UNROLL, CONTRACT, порядок паттернов
> **Эффект**: Делает векторизацию явной и готовой для железа

**Что делает**: Трансформирует оптимизационные примитивы UNROLL/UPCAST в явные операции.

**Зачем это нужно**: UPCAST и UNROLL помечают намерение — что мы хотим сделать. Эта стадия делает это намерение явным, чтобы железо могло его реально выполнить.

**Паттерн**: `symbolic_simple() + pm_pre_expander + pm_group_for_reduce + expander`

Примечание: Morok использует `symbolic_simple()` (не `sym`) на этой стадии, поскольку `symbolic()` уже отработал на стадии 4. Tinygrad использует `sym`, который включает дополнительные паттерны.

**Важно: приоритет паттернов**

Паттерны объединяются и выполняются до fixpoint. Порядок влияет на то, какой паттерн пробуется первым, когда подходят несколько:
1. `sym` первым (символическое упрощение)
2. `pm_pre_expander` вторым (преобразование UPCAST/UNROLL RANGE)
3. `pm_group_for_reduce` третьим (обработка оси GROUP_REDUCE)
4. `expander` последним (основное расширение)

Неправильный приоритет может привести к некорректной векторизации или скоупингу редукций.

**UNROLL и CONTRACT**:

UNROLL и CONTRACT работают в связке:

```text
UNROLL: "Take this one thing and make N copies for different positions"
Example:  x → [x_0, x_1, x_2, x_3]

CONTRACT: "Take these N things and combine them back"
Example:  [a, b, c, d] → one vector containing all four
```

Вместе: UPCAST помечает намерение векторизовать → UNROLL расширяет → CONTRACT объединяет.

**UPCAST range → VECTORIZE**:
```text
// Before: UPCAST marks vectorization intent
RANGE(end=4, UPCAST)
      ↓ [pm_pre_expander]
// Step 1: Convert to UNROLL with constant indices
UNROLL(VCONST([0, 1, 2, 3]))
      ↓ [expander]
// Step 2: Expand operations with UNROLL sources
// Operations now have unrolled sources
      ↓ [CONTRACT or implicit]
// After: explicit VECTORIZE
VECTORIZE(op[0], op[1], op[2], op[3])
```

**UNROLL range → дублированные операции**:

Когда мы говорим «операции дублируются», это звучит как copy-paste. Но на самом деле всё не так. Компилятор создаёт одну SIMD-инструкцию, которая обрабатывает все N элементов одновременно. Представьте SIMD-регистр как коробку, вмещающую 4 числа; сложение двух коробок складывает все 8 чисел за раз.

```text
// Before: UPCAST marks vectorization intent
RANGE(end=3, UPCAST)
      ↓ [pm_pre_expander]
// Step 1: Convert to UNROLL
UNROLL(VCONST([0, 1, 2]))
      ↓ [expander]
// Step 2: Operations expand to handle all positions
// After: operations processed together (not duplicated)
UNROLL([op_at_0, op_at_1, op_at_2])
```

**Взаимодействие UNROLL/END/CONTRACT**:
```text
Before: END(STORE(...), [RANGE(UPCAST)])
             ↓ [pm_pre_expander]
Step 1: END(STORE(...), [UNROLL(VCONST([0,1,2,3]))])
             ↓ [expander]
Step 2: END(CONTRACT(STORE(...×4)), [])
```

**Бродкаст через AFTER/END**:
```text
// Broadcast VECTORIZE (all elements identical)
AFTER(VECTORIZE([x, x, x, x]), deps) → VECTORIZE([AFTER(x, deps), AFTER(x, deps), ...])
```

**Обработка GROUP_REDUCE** (`pm_group_for_reduce`):

GROUP_REDUCE — специальный тип оси для тензорных редукций:

```text
// Before: REDUCE with GROUP_REDUCE ranges
REDUCE(src, [range(GROUP_REDUCE)])
           ↓ [pm_group_for_reduce]
// After: Shared memory reduction pattern
1. Track upstream LOCAL ranges
2. BUFFERIZE result with group ranges (AddrSpace.LOCAL)
3. INDEX into buffer with transformed ranges
4. Final REDUCE with axes (range_id+100, AxisType.REDUCE)
```

Это обеспечивает эффективную аккумуляцию через тензорные ядра с использованием shared-памяти.

**Morok**: `expand.rs`

---

## Стадия 10: Добавление локальных буферов

> **Стадия кратко**
>
> **Цель**: Подготовить буферы для быстрой памяти (shared / L1)
> **Ключевые паттерны**: Bufferize с locals, извлечение хинтов
> **Эффект**: Часто используемые данные остаются в быстрой памяти

**Что делает**: Подготавливает буферы для использования локальной памяти и применяет кодогенерационные чистки.

**Зачем это нужно**: **Локальные буферы** = быстрая память рядом с вычислительным блоком:
- GPU: Shared memory (LDS) — в 100 раз быстрее глобальной памяти
- CPU: L1-кэш — в 10 раз быстрее основной памяти

Компилятор перемещает часто используемые данные в локальные буферы — аналогично тому, как важные файлы хранятся на рабочем столе, а не на сетевом диске.

**Паттерн**: `pm_add_buffers_local + rangeify_codegen`

| Трансформация | Назначение |
|---------------|------------|
| `bufferize_to_store` | Конвертация BUFFERIZE с `allow_locals=true` |
| Удаление обёртки CONTIGUOUS | Удаление оптимизационных хинтов перед кодогенерацией |
| Удаление NOOP | Чистка нопов |

**Morok**: `rangeify/patterns.rs`, `rangeify/transforms.rs`, `optimizer/mod.rs`
