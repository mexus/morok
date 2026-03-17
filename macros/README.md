# morok-macros

Procedural macros for the Morok ML compiler.

## `patterns!`

Generates a pattern matcher from declarative rewrite rules. Used by `morok-schedule`
to build the optimization engine.

```rust
use morok_schedule::patterns;

let matcher = patterns! {
    Add[x, @zero] ~> x,
    Mul[x, @one] ~> x,
    Neg(Neg(x)) ~> Rc::clone(x),
};
```

See [`morok-schedule` README](../schedule/README.md) for the full DSL reference.
