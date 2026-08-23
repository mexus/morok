use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Error, Expr, Ident, Result, Token, Type, braced,
    parse::{Parse, ParseStream},
    token::Comma,
};

pub(crate) struct JitWrapper {
    name: Ident,
    model_ty: Type,
    inputs: Vec<Input>,
    vars: Vec<VarDecl>,
    /// Declared output names. Empty = the classic single-output form (the
    /// `build` closure returns one `Tensor`). Non-empty = the `build` closure
    /// returns a tuple of that many `Tensor`s, in this order, each exposed as
    /// its own `output_buffer_at(i)`-backed accessor.
    outputs: Vec<Ident>,
    build_args: Vec<Ident>,
    build_body: TokenStream,
}

struct Input {
    name: Ident,
}

struct VarDecl {
    name: Ident,
    min: Expr,
    max: Expr,
}

impl Parse for JitWrapper {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        let model_ty: Type = content.parse()?;

        let body;
        braced!(body in input);

        let mut inputs = Vec::new();
        let mut vars = Vec::new();
        let mut outputs = Vec::new();
        let mut build_args = Vec::new();
        let mut build_body = None;

        while !body.is_empty() {
            let first: Ident = body.parse()?;

            if first == "build" {
                let args;
                syn::parenthesized!(args in body);
                build_args = args.parse_terminated(Ident::parse, Comma)?.into_iter().collect();

                let block;
                braced!(block in body);
                build_body = Some(block.parse()?);
            } else if first == "vars" {
                let vars_block;
                braced!(vars_block in body);
                while !vars_block.is_empty() {
                    let name: Ident = vars_block.parse()?;
                    vars_block.parse::<Token![:]>()?;

                    let bounds;
                    syn::parenthesized!(bounds in vars_block);
                    let min: Expr = bounds.parse()?;
                    bounds.parse::<Comma>()?;
                    let max: Expr = bounds.parse()?;
                    if !bounds.is_empty() {
                        return Err(Error::new(bounds.span(), "expected bounds as (min, max)"));
                    }

                    vars.push(VarDecl { name, min, max });

                    if vars_block.peek(Comma) {
                        vars_block.parse::<Comma>()?;
                    }
                }
            } else if first == "outputs" {
                let outs_block;
                braced!(outs_block in body);
                while !outs_block.is_empty() {
                    let out_name: Ident = outs_block.parse()?;
                    outputs.push(out_name);
                    if outs_block.peek(Comma) {
                        outs_block.parse::<Comma>()?;
                    }
                }
                // Tolerate a trailing comma after the `outputs { .. }` block so it
                // reads like the input declarations it sits beside.
                if body.peek(Comma) {
                    body.parse::<Comma>()?;
                }
            } else {
                // Accept (and discard) an optional `: Tensor` for DSL clarity;
                // the macro now allocates placeholder buffers from `InputSpec`
                // so the declared type is informational.
                if body.peek(Token![:]) {
                    body.parse::<Token![:]>()?;
                    let _: Type = body.parse()?;
                }
                inputs.push(Input { name: first });
                if body.peek(Comma) {
                    body.parse::<Comma>()?;
                }
            }
        }

        let build_body = build_body.ok_or_else(|| Error::new(name.span(), "missing `build(...) { ... }` block"))?;

        Ok(JitWrapper { name, model_ty, inputs, vars, outputs, build_args, build_body })
    }
}

pub(crate) fn generate(jit: JitWrapper) -> Result<TokenStream> {
    use std::collections::HashSet;

    let name = &jit.name;
    let model_ty = &jit.model_ty;
    let state_name = format_ident!("{}State", name);

    let input_names: Vec<&Ident> = jit.inputs.iter().map(|i| &i.name).collect();
    let var_names: Vec<&Ident> = jit.vars.iter().map(|v| &v.name).collect();
    let var_min_exprs: Vec<&Expr> = jit.vars.iter().map(|v| &v.min).collect();
    let var_max_exprs: Vec<&Expr> = jit.vars.iter().map(|v| &v.max).collect();
    let var_field_names: Vec<Ident> = jit.vars.iter().map(|v| format_ident!("__var_{}", v.name)).collect();
    let input_id_fields: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_idx", i.name)).collect();
    let input_accessor_names: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_mut", i.name)).collect();
    let input_buffer_id_fields: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_buffer_id", i.name)).collect();
    let input_ast_id_locals: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_ast_id", i.name)).collect();
    let input_realized_locals: Vec<Ident> =
        jit.inputs.iter().map(|i| format_ident!("__jit_input_{}", i.name)).collect();

    let build_args = &jit.build_args;
    let build_body = &jit.build_body;

    let input_name_set: HashSet<String> = jit.inputs.iter().map(|i| i.name.to_string()).collect();
    let var_name_set: HashSet<String> = jit.vars.iter().map(|v| v.name.to_string()).collect();

    for var in &jit.vars {
        if input_name_set.contains(&var.name.to_string()) {
            return Err(Error::new(var.name.span(), "variable name conflicts with input name"));
        }
    }

    let output_names: Vec<&Ident> = jit.outputs.iter().collect();
    let multi_output = !output_names.is_empty();
    let n_outputs = output_names.len();

    for out in &output_names {
        let out_str = out.to_string();
        if input_name_set.contains(&out_str) || var_name_set.contains(&out_str) {
            return Err(Error::new(out.span(), "output name conflicts with an input or variable name"));
        }
    }

    for arg in build_args {
        let arg_name = arg.to_string();
        if !input_name_set.contains(&arg_name) && !var_name_set.contains(&arg_name) {
            return Err(Error::new(arg.span(), "build arg must match an input or a declared variable"));
        }
    }

    let build_arg_sources: Vec<TokenStream> = build_args.iter().map(|arg| quote! { #arg }).collect();

    let prepare_params: Vec<TokenStream> =
        input_names.iter().map(|n| quote! { #n: svod_model::jit::InputSpec }).collect();

    let var_inits =
        var_names.iter().zip(var_field_names.iter()).zip(var_min_exprs.iter().zip(var_max_exprs.iter())).map(
            |((var_name, field_name), (min_expr, max_expr))| {
                quote! {
                    let #field_name = svod_tensor::Variable::new(
                        stringify!(#var_name),
                        (#min_expr) as i64,
                        (#max_expr) as i64,
                    );
                }
            },
        );

    // For each declared `vars { name: (min, max), ... }` entry, emit three
    // builders:
    //   * `with_<name>_bound(max)`     — override only the upper bound
    //   * `with_<name>_min_bound(min)` — override only the lower bound
    //   * `with_<name>_fixed(value)`   — pin both bounds to one value, making
    //     the variable a JIT-time constant (specializable kernels, single
    //     valid value at execute time)
    //
    // All three panic if the resulting `[min, max]` is empty so misuse fails
    // loud at construction instead of at bind/execute time. Variable names
    // are checked at compile time via the generated method names. Must be
    // chained before `prepare` — the JIT plan captures the bounds when the
    // build closure runs.
    let with_var_bound_methods = var_names.iter().zip(var_field_names.iter()).flat_map(|(var_name, field_name)| {
        let max_setter = format_ident!("with_{}_bound", var_name);
        let min_setter = format_ident!("with_{}_min_bound", var_name);
        let fixed_setter = format_ident!("with_{}_fixed", var_name);
        let max_doc = format!(
            "Override the upper bound for the `{var_name}` symbolic variable. \
             Must be called before `prepare`/`prepare_with_config`. Panics if \
             `max < min`."
        );
        let min_doc = format!(
            "Override the lower bound for the `{var_name}` symbolic variable. \
             Must be called before `prepare`/`prepare_with_config`. Panics if \
             `min > max`."
        );
        let fixed_doc = format!(
            "Pin `{var_name}` to a single value, making it a JIT-time \
             constant. Sets both bounds to `value` so only `value` is \
             accepted at execute time. Must be called before \
             `prepare`/`prepare_with_config`. Panics on `value == 0`."
        );
        let name_str = format!("{var_name}");
        std::iter::empty()
            .chain(std::iter::once(quote! {
                #[doc = #max_doc]
                pub fn #max_setter(mut self, max: usize) -> Self {
                    let (min, _) = self.#field_name.bounds();
                    let max_i64 = max as i64;
                    assert!(
                        max_i64 >= min,
                        "{}: with_{}_bound({max}) creates empty range (min={min})",
                        #name_str, #name_str,
                    );
                    self.#field_name = svod_tensor::Variable::new(stringify!(#var_name), min, max_i64);
                    self
                }
            }))
            .chain(std::iter::once(quote! {
                #[doc = #min_doc]
                pub fn #min_setter(mut self, min: usize) -> Self {
                    let (_, max) = self.#field_name.bounds();
                    let min_i64 = min as i64;
                    assert!(
                        min_i64 <= max,
                        "{}: with_{}_min_bound({min}) exceeds upper bound max={max}",
                        #name_str, #name_str,
                    );
                    self.#field_name = svod_tensor::Variable::new(stringify!(#var_name), min_i64, max);
                    self
                }
            }))
            .chain(std::iter::once(quote! {
                #[doc = #fixed_doc]
                pub fn #fixed_setter(mut self, value: usize) -> Self {
                    assert!(value > 0, "{}: with_{}_fixed(0) is not allowed", #name_str, #name_str);
                    let v = value as i64;
                    self.#field_name = svod_tensor::Variable::new(stringify!(#var_name), v, v);
                    self
                }
            }))
    });

    let prepare_var_bindings = var_names.iter().zip(var_field_names.iter()).map(|(var_name, field_name)| {
        quote! {
            let #var_name = self.#field_name
                .bind(self.#field_name.bounds().1)
                .map_err(|e| svod_model::jit::JitError::Tensor { source: Box::new(e) })?;
        }
    });

    let build_closure = quote! {
        (|| {
            let model = &self.model;
            let (#(#build_args),*) = (#(#build_arg_sources),*);
            #build_body
        })()
    };

    let input_realizations = input_names.iter().zip(input_realized_locals.iter()).map(|(input_name, local)| {
        quote! {
            let #local = if #input_name.device_local {
                // Eager device-local zeros: no host mapping, init staged
                // through the copy engine. Skips the realize schedule.
                let numel: usize = #input_name.shape.iter().product();
                svod_tensor::Tensor::from_bytes_shaped_spec(
                    &vec![0u8; numel * #input_name.dtype.bytes()],
                    &#input_name.shape,
                    #input_name.dtype.clone(),
                    svod_dtype::default_device::default_device(),
                    svod_device::BufferSpec { cpu_access: false, ..Default::default() },
                )
            } else {
                let mut t = svod_tensor::Tensor::zeros(&#input_name.shape, #input_name.dtype.clone())
                    .map_err(|e| svod_model::jit::JitError::Tensor { source: Box::new(e) })?;
                // Inputs are host-written every execute (`as_array_mut` pack):
                // the plan-level `device_local_outputs` opt-in must not leak
                // into this realization or the buffer loses its host mapping.
                let mut input_config = config.clone();
                input_config.device_local_outputs = false;
                t.realize_with(&input_config)
                    .map_err(|e| svod_model::jit::JitError::Tensor { source: Box::new(e) })?;
                t
            };
            let #input_name = &#local;
        }
    });

    let buffer_id_extractions =
        input_names.iter().zip(input_buffer_id_fields.iter()).zip(input_ast_id_locals.iter()).map(
            |((input_name, buf_field), ast_field)| {
                quote! {
                    let #buf_field = #input_name.buffer().ok_or(svod_model::jit::JitError::NotPrepared)?.id();
                    let #ast_field = #input_name.uop().id;
                }
            },
        );

    let duplicate_input_checks = input_names.iter().zip(input_buffer_id_fields.iter()).enumerate().flat_map(
        |(left_idx, (left_name, left_buf_field))| {
            input_names.iter().zip(input_buffer_id_fields.iter()).skip(left_idx + 1).map(
                move |(right_name, right_buf_field)| {
                    let left_name_str = left_name.to_string();
                    let right_name_str = right_name.to_string();
                    quote! {
                        if #left_buf_field == #right_buf_field {
                            return Err(svod_model::jit::JitError::DuplicateInputBuffer {
                                name: #right_name_str,
                                duplicate_of: #left_name_str,
                                buffer_id: #right_buf_field,
                            });
                        }
                    }
                },
            )
        },
    );

    let index_resolution =
        input_id_fields.iter().zip(input_buffer_id_fields.iter()).zip(input_ast_id_locals.iter()).map(
            |((idx_field, buf_id_field), ast_id_field)| {
                quote! {
                    let #idx_field = plan
                        .ast_to_buffer_map()
                        .get(&#ast_id_field)
                        .copied()
                        .or_else(|| plan.buffers().iter().position(|b| b.id() == #buf_id_field));
                }
            },
        );

    let idx_fields: Vec<&Ident> = input_id_fields.iter().collect();
    let buf_id_fields: Vec<&Ident> = input_buffer_id_fields.iter().collect();
    let state_init = quote! {
        #state_name {
            plan,
            #( #idx_fields, )*
            #( #buf_id_fields, )*
        }
    };

    let accessor_impls = input_accessor_names
        .iter()
        .zip(input_id_fields.iter())
        .zip(input_buffer_id_fields.iter())
        .zip(input_names.iter())
        .map(|(((accessor, idx_field), buf_id_field), input_name)| {
            let name_str = input_name.to_string();
            quote! {
                pub fn #accessor(&mut self) -> svod_model::jit::Result<&mut svod_device::Buffer> {
                    let state = self.state.as_mut().ok_or(svod_model::jit::JitError::NotPrepared)?;
                    let idx = match state.#idx_field {
                        Some(idx) => idx,
                        None => {
                            let idx = state
                                .plan
                                .buffers()
                                .iter()
                                .position(|b| b.id() == state.#buf_id_field)
                                .ok_or(svod_model::jit::JitError::InputBufferNotFound { name: #name_str })?;
                            state.#idx_field = Some(idx);
                            idx
                        }
                    };
                    state.plan.buffer_at_mut(idx)
                        .ok_or(svod_model::jit::JitError::InputBufferNotFound { name: #name_str })
                }
            }
        });

    // Per-input on-device copy helpers: copy a region of declared output
    // `out_pos` into the input's buffer with NO host round-trip (the plan owns
    // both buffers; the split borrow lives in the runtime). Used to recycle
    // recurrent state output→input.
    let copy_helper_impls = input_accessor_names
        .iter()
        .zip(input_id_fields.iter())
        .zip(input_buffer_id_fields.iter())
        .zip(input_names.iter())
        .map(|(((_accessor, idx_field), buf_id_field), input_name)| {
            let helper = format_ident!("copy_output_to_{}", input_name);
            let name_str = input_name.to_string();
            quote! {
                pub fn #helper(
                    &mut self,
                    out_pos: usize,
                    dst_off: usize,
                    src_off: usize,
                    len: usize,
                ) -> svod_model::jit::Result<()> {
                    let state = self.state.as_mut().ok_or(svod_model::jit::JitError::NotPrepared)?;
                    let idx = match state.#idx_field {
                        Some(idx) => idx,
                        None => {
                            let idx = state
                                .plan
                                .buffers()
                                .iter()
                                .position(|b| b.id() == state.#buf_id_field)
                                .ok_or(svod_model::jit::JitError::InputBufferNotFound { name: #name_str })?;
                            state.#idx_field = Some(idx);
                            idx
                        }
                    };
                    state.plan.copy_output_region_to_buffer(out_pos, idx, dst_off, src_off, len)
                        .map_err(|e| svod_model::jit::JitError::Runtime { source: e })
                }
            }
        });

    // Build the output tensor(s) and compile the plan. The single-output form
    // (no `outputs` clause) keeps the original `output: Tensor` codegen verbatim;
    // the multi-output form destructures the build closure's tuple in declared
    // order and feeds all of them to `prepare_batch_with` (which preserves order),
    // then asserts the plan kept exactly that many outputs.
    let build_and_compile = if multi_output {
        quote! {
            let (#(#output_names,)*) = #build_closure
                .map_err(|e| svod_model::jit::JitError::Build { source: Box::new(e) as _ })?;
            let mut __jit_outputs: [svod_tensor::Tensor; #n_outputs] = [#(#output_names,)*];
            let plan = svod_tensor::Tensor::prepare_batch_with(__jit_outputs.iter_mut(), config)
                .map_err(|e| svod_model::jit::JitError::Tensor { source: Box::new(e) })?;
            if plan.num_outputs() != #n_outputs {
                return Err(svod_model::jit::JitError::OutputCountMismatch {
                    declared: #n_outputs,
                    actual: plan.num_outputs(),
                });
            }
        }
    } else {
        quote! {
            let output: svod_tensor::Tensor = #build_closure
                .map_err(|e| svod_model::jit::JitError::Build { source: Box::new(e) as _ })?;
            let mut output = output;
            let plan = svod_tensor::Tensor::prepare_batch_with(std::iter::once(&mut output), config)
                .map_err(|e| svod_model::jit::JitError::Tensor { source: Box::new(e) })?;
        }
    };

    // One accessor per declared output, backed by positional `output_buffer_at(i)`
    // (i = declared order = `prepare_batch_with` order). Empty for single-output.
    let output_named_accessors = output_names.iter().enumerate().map(|(i, out_name)| {
        quote! {
            pub fn #out_name(&self) -> svod_model::jit::Result<&svod_device::Buffer> {
                let state = self.state.as_ref().ok_or(svod_model::jit::JitError::NotPrepared)?;
                state.plan.output_buffer_at(#i).ok_or(svod_model::jit::JitError::NotPrepared)
            }
        }
    });

    let expanded = quote! {
        pub struct #name {
            model: #model_ty,
            state: Option<#state_name>,
            #( #var_field_names: svod_tensor::Variable, )*
        }

        struct #state_name {
            plan: svod_runtime::ExecutionPlan,
            #( #input_id_fields: Option<usize>, )*
            #( #input_buffer_id_fields: svod_device::BufferId, )*
        }

        impl #name {
            pub fn new(model: #model_ty) -> Self {
                #(#var_inits)*
                Self {
                    model,
                    state: None,
                    #( #var_field_names, )*
                }
            }

            #(#with_var_bound_methods)*

            pub fn prepare(&mut self, #(#prepare_params),*) -> svod_model::jit::Result<()> {
                let config = svod_tensor::PrepareConfig::from_env();
                self.prepare_with_config(#(#input_names,)* &config)
            }

            pub fn prepare_with_config(
                &mut self,
                #(#prepare_params,)*
                config: &svod_tensor::PrepareConfig,
            ) -> svod_model::jit::Result<()> {
                #(#input_realizations)*
                #(#buffer_id_extractions)*
                #(#duplicate_input_checks)*

                #(#prepare_var_bindings)*

                #build_and_compile

                #(#index_resolution)*

                self.state = Some(#state_init);
                Ok(())
            }

            #(#accessor_impls)*

            pub fn output(&self) -> svod_model::jit::Result<&svod_device::Buffer> {
                let state = self.state.as_ref().ok_or(svod_model::jit::JitError::NotPrepared)?;
                state.plan.output_buffer().ok_or(svod_model::jit::JitError::NotPrepared)
            }

            #(#output_named_accessors)*

            #(#copy_helper_impls)*

            pub fn buffers(&self) -> svod_model::jit::Result<&[svod_device::Buffer]> {
                let state = self.state.as_ref().ok_or(svod_model::jit::JitError::NotPrepared)?;
                Ok(state.plan.buffers())
            }

            pub fn output_buffers(&self) -> svod_model::jit::Result<Vec<&svod_device::Buffer>> {
                let state = self.state.as_ref().ok_or(svod_model::jit::JitError::NotPrepared)?;
                Ok(state.plan.output_buffers())
            }

            pub fn input_buffer_ids(&self) -> svod_model::jit::Result<Vec<svod_device::BufferId>> {
                let state = self.state.as_ref().ok_or(svod_model::jit::JitError::NotPrepared)?;
                Ok(vec![#( state.#input_buffer_id_fields ),*])
            }

            pub fn prepared_kernels(&self) -> svod_model::jit::Result<Vec<&svod_runtime::PreparedKernel>> {
                let state = self.state.as_ref().ok_or(svod_model::jit::JitError::NotPrepared)?;
                Ok(state.plan.prepared_kernels())
            }

            pub fn execute(&mut self) -> svod_model::jit::Result<()> {
                let state = self.state.as_mut().ok_or(svod_model::jit::JitError::NotPrepared)?;
                state.plan.execute()
                    .map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }

            pub fn execute_profiled(&mut self) -> svod_model::jit::Result<Vec<svod_runtime::KernelProfile>> {
                let state = self.state.as_mut().ok_or(svod_model::jit::JitError::NotPrepared)?;
                state.plan.execute_profiled()
                    .map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }

            pub fn execute_profiled_static(&mut self) -> svod_model::jit::Result<Vec<svod_runtime::KernelProfile>> {
                let state = self.state.as_mut().ok_or(svod_model::jit::JitError::NotPrepared)?;
                state.plan.profile(&svod_runtime::ProfileOptions::default())
                    .map(|mut profile| profile.stages.pop().map_or_else(Vec::new, |stage| stage.kernels))
                    .map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }

            pub fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> svod_model::jit::Result<()> {
                let state = self.state.as_mut().ok_or(svod_model::jit::JitError::NotPrepared)?;
                state.plan.execute_with_vars(vars)
                    .map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }

            pub fn execute_with_vars_profiled(
                &mut self,
                vars: &[(&str, i64)],
            ) -> svod_model::jit::Result<Vec<svod_runtime::KernelProfile>> {
                let state = self.state.as_mut().ok_or(svod_model::jit::JitError::NotPrepared)?;
                state.plan.execute_with_vars_profiled(vars)
                    .map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }
        }
    };

    Ok(expanded)
}
