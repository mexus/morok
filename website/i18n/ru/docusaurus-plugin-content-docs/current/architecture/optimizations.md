---
sidebar_label: Система оптимизаций
---

# Оптимизации на основе паттернов

Откройте любой продакшн ML-компилятор — и найдёте десятки оптимизационных проходов: свёртка констант, удаление мёртвого кода, фьюзинг операторов, тайлинг циклов, векторизация, оптимизация раскладки памяти. У каждого прохода свои структуры данных, своя логика обхода, свои баги.

Morok использует другой подход: **один механизм для всего**.

```text
Traditional Compiler:              Morok:
┌─────────────────────────┐       ┌─────────────────────────┐
│  Constant Folding       │       │                         │
│  Dead Code Elimination  │       │   patterns! {           │
│  Loop Unrolling         │       │       Add[x, @zero] ~> x│
│  Operator Fusion        │       │       Mul[x, @zero] ~> 0│
│  Vectorization          │       │       // ...more        │
│  Memory Planning        │       │   }                     │
│  ...20 more passes      │       │                         │
└─────────────────────────┘       │   graph_rewrite(...)    │
     Custom logic each            └─────────────────────────┘
                                       One mechanism
```

Каждая оптимизация в Morok выражается как **паттерн**: «когда видишь эту структуру, замени её вот этой». Одна и та же функция `graph_rewrite()` применяет свёртку констант, преобразует movement-операции в циклы, оптимизирует паттерны доступа к памяти и снижает до аппаратных примитивов.

Эта глава объясняет, как работают оптимизации на основе паттернов и почему они мощные.

---

## DSL `patterns!`

Morok предоставляет предметно-ориентированный язык для написания оптимизационных паттернов. Вот как он выглядит:

```rust
patterns! {
    // Identity folding: x + 0 → x
    Add[x, @zero] ~> |x| x.clone(),

    // Constant folding: 3 + 4 → 7
    Add(a @const(a_val), b @const(b_val))
        => |a, a_val, b_val| eval_add(a_val, b_val).map(|r| UOp::const_(a.dtype(), r)),

    // Self-folding: x / x → 1
    Idiv(x, x) ~> |x| UOp::one(x.dtype()),

    // Dead code elimination: if(true) { t } else { f } → t
    Where(@true, t, _f) ~> |t| t.clone(),
}
```

Макрос компилирует эти паттерны в эффективный Rust-код. Разберём синтаксис:

| Синтаксис | Значение | Пример |
|-----------|----------|--------|
| `(x, y)` | **Упорядочено.** Сопоставляется в точном порядке. | `Sub(x, @zero) ~> x` |
| `[x, y]` | **Коммутативно.** Пробуются оба порядка. | `Add[x, @zero] ~> x` |
| `@zero` | **Нулевая константа.** Совпадает с 0 или 0.0. | `Mul[_, z @ @zero] ~> z` |
| `@one` | **Единичная константа.** Совпадает с 1 или 1.0. | `Mul[x, @one] ~> x` |
| `@const(val)` | **Извлечение константы.** Связывает значение. | `Add(@const(a), @const(b))` |
| `x, x` | **Один и тот же операнд.** Автогенерируется проверка ptr_eq. | `Idiv(x, x) ~> UOp::one(...)` |
| `~>` | **Безусловный.** Всегда успешен, возвращает `Arc<UOp>`. | `Add[x, @zero] ~> x` |
| `=>` | **Условный.** Может не сработать, возвращает `Option<Arc<UOp>>`. | `=> eval(...).map(...)` |
| `for op in binary [...]` | **Шаблон.** Генерация паттернов для нескольких операций. | См. ниже |
| `@context Type` | **С состоянием.** Доступ к мутабельному контексту в паттернах. | См. ниже |

### Раскрытие шаблонов

Вместо написания одного и того же паттерна для каждой бинарной операции используйте for-цикл:

```rust
patterns! {
    for op in binary [Add, Mul, Sub, Idiv, Fdiv, Max] {
        op(a @const(a_val), b @const(b_val))
            => |a, a_val, b_val| eval_binary(op, a_val, b_val)
                .map(|r| UOp::const_(a.dtype(), r))
    }
}
```

Это раскрывается в шесть отдельных паттернов во время компиляции — по одному для каждой операции.

### Паттерны с состоянием

Некоторым оптимизациям нужен контекст (например, в каком ядре мы находимся, какие диапазоны активны). Объявите тип контекста:

```rust
patterns! {
    @context KernelContext;

    ReduceAxis { src } => |reduce, src, ctx| {
        ctx.record_reduction(reduce);
        transform_reduce(reduce, src, ctx)
    }
}
```

Контекст передаётся последним аргументом в замыкания паттернов.

---

## Как работает сопоставление паттернов

Макрос `patterns!` генерирует `SimplifiedPatternMatcher`, который диспатчит паттерны за **O(1)**.

### Индекс OpKey

У каждого UOp есть тип операции (Add, Mul, Load и т.д.). Макрос `#[derive(PatternEnum)]` генерирует enum `OpKey`, отображающий операции в хэшируемые ключи:

```rust
pub enum OpKey {
    Binary(BinaryOp),    // Add, Mul, Sub, ...
    Unary(UnaryOp),      // Neg, Sqrt, Exp, ...
    Ternary(TernaryOp),  // Where, MulAcc
    Const,
    Load,
    Store,
    // ... one variant per operation category
}
```

### Структура Matcher

```rust
pub struct SimplifiedPatternMatcher<C = ()> {
    indexed: HashMap<OpKey, Vec<PatternClosure<C>>>,  // O(1) lookup
    wildcards: Vec<PatternClosure<C>>,                 // patterns matching any op
}
```

При сопоставлении UOp:

1. **Извлекаем OpKey** из операции UOp
2. **Ищем** в HashMap — O(1)
3. **Пробуем каждое замыкание**, пока одно не сработает
4. **Откатываемся** на wildcards, если ни один индексированный паттерн не совпал

Это в 5–10 раз быстрее линейного перебора всех паттернов.

### Обработка коммутативности

Для паттернов вроде `Add[x, @zero]` макрос генерирует код, пробующий оба порядка:

```rust
// Try (x, @zero)
if let Some(result) = try_match_ordered(&children[0], &children[1]) {
    return result;
}
// Try (@zero, x)
if let Some(result) = try_match_ordered(&children[1], &children[0]) {
    return result;
}
```

### Обнаружение дубликатов

Когда вы пишете `Idiv(x, x)`, паттерн должен сработать только если оба операнда — *один и тот же* UOp (равенство указателей, а не структурное). Макрос автоматически генерирует эту проверку:

```rust
// Generated code for Idiv(x, x)
let x = &children[0];
let x_dup = &children[1];
if !Arc::ptr_eq(x, x_dup) {
    return NoMatch;
}
// ... rest of pattern
```

Это использует hash consing — идентичные подвыражения разделяют один указатель.

---

## Движок перезаписи: двухстадийный алгоритм

> **Примечание:** Это упрощённое описание. Реальный движок использует трёхстадийный стековый алгоритм с path compression для эффективности.

Одного сопоставления паттернов недостаточно. Рассмотрим выражение:

```text
WHERE(Lt(3, 5), t, f)
```

Чтобы его упростить, нужны два шага:
1. `Lt(3, 5)` → `true` (свёртка констант)
2. `WHERE(true, t, f)` → `t` (удаление мёртвого кода)

Но паттерн `WHERE` не сработает, пока его дочерний узел не упрощён. Движок перезаписи решает это **двухстадийным алгоритмом**.

### Стадия 0: Применение паттернов

```rust
fn rewrite_stage0(&mut self, uop: &Arc<UOp>) -> RewriteResult {
    match self.matcher.try_match(uop) {
        Some(replacement) => RewriteResult::Rewritten(replacement),
        None => RewriteResult::Gate(uop.clone()),  // process children
    }
}
```

Если ни один паттерн не совпал, возвращаем `Gate` — сигнал сначала обработать дочерние узлы.

### Стадия 1: Реконструкция

После перезаписи дочерних узлов перестраиваем узел с новыми потомками и снова пробуем паттерны:

```rust
fn rewrite_stage1(&mut self, uop: &Arc<UOp>, new_children: Vec<Arc<UOp>>) {
    // Rebuild with optimized children
    let rebuilt = uop.with_sources(new_children);

    // Try patterns again—might match now!
    match self.matcher.try_match(&rebuilt) {
        Some(replacement) => replacement,
        None => rebuilt,
    }
}
```

### Магия: каскадные оптимизации

```text
Stage 0: WHERE(Lt(3, 5), t, f)     → Gate (no match, process children)
         └── Lt(3, 5)              → true (constant folding matches!)

Stage 1: WHERE(true, t, f)         → t (dead code elimination matches!)
```

Стадия реконструкции повторно применяет паттерны, что позволяет многошаговым оптимизациям сработать за один обход.

### Ограничения безопасности

Для предотвращения бесконечных циклов в движке есть лимиты:
- **1000 итераций** максимум на узел
- **500 000 итераций** максимум в сумме
- Panic с диагностикой при превышении лимитов

На практике корректные паттерны сходятся быстро.

---

## Полный пайплайн оптимизаций

Сопоставление паттернов — часть более крупного пайплайна. При вызове `tensor.realize()` происходит следующее:

```text
Tensor.realize()
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  RANGEIFY                                               │
│  Convert movement ops (RESHAPE, PERMUTE, EXPAND)        │
│  into explicit RANGE loops with INDEX operations        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  KERNEL SPLITTING                                       │
│  Split computation graph at STORE boundaries            │
│  Each STORE becomes a separate kernel                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  FOR EACH KERNEL:                                       │
│                                                         │
│  1. Symbolic Simplification (algebraic patterns)        │
│                                                         │
│  2. Scheduler Creation                                  │
│     └── Convert LOOP → GLOBAL for GPU parallelization   │
│                                                         │
│  3. Kernel Optimization (heuristic OR beam search)      │
│     ├── Tensor Cores (WMMA) for matmul                  │
│     ├── Vectorization (UPCAST)                          │
│     ├── Loop Unrolling (UNROLL)                         │
│     ├── GPU Local Memory (LOCAL)                        │
│     ├── Grouped Reductions (GROUP)                      │
│     └── Threading (THREAD) for CPU                      │
│                                                         │
│  4. Post-Optimization Passes                            │
│     ├── Devectorize (memory coalescing)                 │
│     ├── Expand (UNROLL → vector operations)             │
│     ├── FMA Decomposition (a*b+c → MulAcc)              │
│     └── Bool Storage (cast bool↔uint8 for memory)       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  CODE GENERATION                                        │
│  Render optimized AST to LLVM IR, compile, execute      │
└─────────────────────────────────────────────────────────┘
```

Каждый блок использует перезапись на основе паттернов. Разница — в том, какие паттерны применяются:

- **Rangeify**: Movement-операции → паттерны BUFFERIZE + INDEX
- **Символьные**: Паттерны алгебраического упрощения
- **Пост-оптимизация**: Паттерны оптимизации доступа к памяти

---

## Оптимизация ядер: эвристики vs beam search

После символьного упрощения каждому ядру нужны *решения по планированию*: как тайлить циклы, где параллелизовать, использовать ли tensor cores. Morok предлагает две стратегии.

### Эвристики (по умолчанию)

Эвристический оптимизатор применяет оптимизации в фиксированном порядке:

```rust
pub fn hand_coded_optimizations(scheduler: &mut Scheduler) {
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

### Beam search (опционально)

Для продакшн-нагрузок beam search находит лучшие расписания:

```rust
pub fn beam_search(scheduler: Scheduler, config: BeamConfig) -> Scheduler {
    let mut beam = vec![scheduler];
    let deadline = Instant::now() + config.time_limit;

    while Instant::now() < deadline {
        let mut candidates = vec![];

        for state in &beam {
            // Generate all valid next actions
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

Пространство действий включает ~162 базовых действия (зависит от доступного параллелизма):
- `UPCAST(axis, amount)` — векторизация размерности выхода
- `UNROLL(axis, amount)` — развёртка цикла редукции
- `LOCAL(axis, amount)` — использование GPU shared memory
- `GROUP(axis, amount)` — двухстадийная редукция
- `GROUPTOP(axis, amount)` — grouped reduction для tensor cores
- `THREAD(axis, amount)` — CPU-параллелизация
- `SWAP(axis1, axis2)` — перестановка глобальных размерностей

**Плюсы**: Находит близкие к оптимальным расписания, адаптируется к железу.

**Минусы**: Минуты на ядро (но результаты кэшируются по хэшу AST).

### Конфигурация

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

Разные ML-компиляторы используют разные подходы к оптимизации:

| Аспект | XLA | TVM/Ansor | Triton | **Morok** |
|--------|-----|-----------|--------|-----------|
| **Философия** | Фиксированные эвристики | Поиск | Управление программистом | На основе паттернов |
| **Фьюзинг** | Консервативные правила | Tile-and-fuse | Block-level | Перезапись графа |
| **Автотюнинг** | Нет | Эволюционный + cost model | Grid search | Beam search |
| **Стоимость тюнинга** | 0 | Часы | Минуты | Минуты (кэшируется) |
| **Гибкость** | Низкая | Высокая | Средняя | Высокая |
| **Прозрачность** | Низкая (C++-проходы) | Средняя (Python) | Средняя (DSL) | Высокая (patterns!) |

### XLA — продакшн-консерватизм

XLA использует фиксированные эвристики для решений по фьюзингу. Безопасно и предсказуемо, но оставляет производительность на столе. Правила фьюзинга захардкожены в C++ — для их расширения нужно глубокое знание компилятора.

### TVM/Ansor — максимальный автотюнинг

TVM разделяет *что* вычислять и *как* вычислять. Ansor использует эволюционный поиск с обучаемой cost model для исследования пространства расписаний. Может достигать лучшей в классе производительности, но тюнинг занимает часы на модель.

### Triton — управление программистом

Triton предоставляет Python-подобный DSL, где вы явно пишете блочные алгоритмы. Компилятор занимается аллокацией регистров и управлением памятью. Хороший баланс контроля и автоматизации, но требует экспертизы в GPU-программировании.

### Morok — композиция паттернов

Идея Morok: выражать оптимизации как компонуемые паттерны. Каждый паттерн локален и верифицируем. Сложные оптимизации возникают из композиции. Beam search добавляет автотюнинг при необходимости, с кэшированием результатов для повторного использования.

---

## Почему это важно: практическая польза

Оптимизации на основе паттернов дают конкретные преимущества для разработчиков:

**Отладка прямая.** Паттерны — это читаемый код. Добавьте `println!` в любой паттерн, чтобы отследить, когда он срабатывает:

```rust
patterns! {
    Add[x, @zero] ~> |x| {
        println!("Folding add-zero: {:?}", x);
        x.clone()
    }
}
```

**Расширяемость простая.** Добавление своей оптимизации — пара строк:

```rust
patterns! {
    // Your domain-specific optimization
    MyOp(x, y) if is_special_case(x, y) ~> transform(x, y)
}
```

Не нужно разбираться во внутренностях компилятора, писать визиторы или модифицировать pass manager.

**Корректность локальна.** Каждый паттерн — маленькая теорема: «если появляется эта структура, замена на ту структуру сохраняет семантику». Каждый паттерн верифицируется независимо. Композиция корректных паттернов даёт корректные программы.

**Производительность настраиваема.** O(1) диспатч паттернов быстр по умолчанию. Включите beam search для продакшн-нагрузок. Кэшируйте результаты по хэшу AST — тюним один раз, пользуемся всегда.

---

## Глубинная идея

Сопоставление паттернов обменивает общность на компонуемость.

Универсальный оптимизационный проход может делать что угодно — и в этом проблема. Его трудно верифицировать, трудно расширять, трудно компоновать с другими проходами. Порядок важен. Взаимодействия тонки.

Паттерн ограничен: он сопоставляет конкретную структуру и порождает конкретную замену. Но ограничения дают компонуемость. Запускайте паттерны в любом порядке — результат сходится к одной и той же фиксированной точке. Добавляйте новые паттерны, не ломая существующие. Удаляйте паттерны без каскадных сбоев.

Каждый паттерн — теорема о семантической эквивалентности. Движок перезаписи — доказыватель теорем, находящий вывод от входа к оптимизированному выходу. Корректность следует из корректности отдельных шагов.

Это философия Unix, применённая к компиляторам: маленькие, сфокусированные инструменты, которые компонуются. Оптимизации на основе паттернов не решат все задачи — но те, что решают, решают элегантно.
