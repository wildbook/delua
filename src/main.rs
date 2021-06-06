#![feature(array_windows)]
#![feature(map_first_last)]

use std::{
    collections::HashMap,
    io::{BufWriter, Cursor, Write},
};

use itertools::Itertools;

use crate::stage_2::{Arg, ArgDir, ArgSlot, OpInfo};

pub mod stage_1;
pub mod stage_2;
pub mod stage_3;
mod util;

fn escape(s: String) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', r"\n")
        .replace('|', r"\|")
        .replace('<', r"\<")
        .replace('>', r"\>")
        .replace('{', r"\{")
        .replace('}', r"\}")
        .replace('"', r#"\""#)
        .replace(r"\\l", r"\l")
}

fn output_stage_1(top: stage_2::Function) {
    let mut stack = vec![top.clone()];
    while let Some(func) = stack.pop() {
        let block = func.blocks();
        log::trace!("|blocks");

        for (_, b) in block.iter().sorted_by_key(|(&a, _)| a) {
            log::trace!("{:#}", b);
        }

        stack.extend(func.prototypes);
    }
}

fn output_stage_2(
    top: stage_2::Function,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    let mut out = BufWriter::new(std::fs::File::create("output.dot")?);
    writeln!(out, "digraph L {{")?;

    let mut i = 0;
    let mut stack = vec![top.clone()];
    while let Some(func) = stack.pop() {
        writeln!(out, "subgraph fn_{} {{", i)?;
        writeln!(out, "  node [shape=record fontname=Consolas];")?;

        let blocks = func.blocks();
        for (a, block) in blocks {
            let mut code = String::new();
            let mut debug = String::new();
            for instr in block.instructions.iter() {
                code += &format!("{}\\l", instr);
                debug += &format!("{:?}\\l", instr);
            }

            code = format!(
                "  fn_{}_{} [label=\"{}\", tooltip=\"{}\"]",
                i,
                a,
                escape(code),
                escape(debug)
            );

            writeln!(out, "{}", code)?;
            for t in block.forks_to.iter() {
                write!(out, "fn_{}_{} -> fn_{}_{} ", i, block.addr_start, i, t.addr)?;
                match block.addr_start == t.addr {
                    true => writeln!(out, "[dir=back]")?,
                    false => writeln!(out)?,
                }
            }
        }

        writeln!(out, "}}")?;

        stack.extend(func.prototypes);
        i += 1;
    }

    writeln!(out, "}}")?;
    out.flush()?;
    drop(out);

    Ok(std::process::Command::new("dot").arg("output.dot").arg("-O").arg("-Tsvg").spawn()?)
}

fn output_stage_3(
    top: stage_2::Function,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    let mut lifted = stage_3::Function::lift(top.clone()).with_name("Entrypoint");
    dbg!(lifted.optimize_until_complete());

    let mut out = BufWriter::new(std::fs::File::create("output_lifted.dot")?);

    writeln!(out, "digraph L {{")?;
    writeln!(out, "  graph [splines=ortho, packmode=\"cluster\"]")?;
    writeln!(out, "  node [shape=record fontname=Consolas style=filled color=white margin=0.2];")?;
    let mut fn_index = 0;
    let mut stack_fn = vec![lifted];
    while let Some(func) = stack_fn.pop() {
        writeln!(out, "subgraph cluster_fn_{} {{", fn_index)?;
        writeln!(out, "  fontname = Consolas")?;
        writeln!(out, "  style = filled")?;

        match func.name {
            Some(name) => writeln!(out, "  label = \"{} (Function {})\"", name, fn_index)?,
            None => writeln!(out, "  label = \"Function {}\"", fn_index)?,
        }

        let blocks = func.blocks;
        let first_block = blocks.first_key_value().map(|(id, _)| *id);

        for (block_id, block) in blocks {
            let mut content = String::new();
            let mut tooltip = String::new();

            let header = format!("ID {}", *block_id);
            content += &format!("{}\n{}\n", header, "─".repeat(header.chars().count()));

            let pad = (block.instructions.len() as f32).log10() as usize + 1;
            for (idx, instr) in block.instructions.into_iter().enumerate() {
                let instr_text = instr.text(&func.constants);
                content += &format!("\u{200B}{: >pad$} | {}\\l", idx, instr_text, pad = pad);
                tooltip += &format!("\u{200B}{: >pad$} | {:?}\\l", idx, instr, pad = pad);
            }

            content = format!(
                "  fn_{}_{} [label=\"{}\", tooltip=\"{}\"]",
                fn_index,
                *block_id,
                escape(content),
                escape(tooltip)
            );

            writeln!(out, "{}", content)?;
            for t in block.forks_to.iter() {
                write!(out, "  fn_{}_{} -> fn_{}_{} ", fn_index, *block_id, fn_index, **t)?;
                match *block_id == **t {
                    true => writeln!(out, "[dir=back]")?,
                    false => writeln!(out)?,
                }
            }
        }

        if !func.children.is_empty() {
            let protos = escape(
                func.children
                    .iter()
                    .enumerate()
                    .map(|(c, v)| format!("{}: {}", c, v.name.as_deref().unwrap_or("✖ unknown ✖")))
                    .join("\\l"),
            );
            writeln!(out, "  protos_{} [label=\"PROTOS\\n──────\\n{}\\l\"]", fn_index, protos)?;

            if let Some(first) = first_block {
                writeln!(out, "protos_{} -> fn_{}_{}", fn_index, fn_index, *first)?;
            }
        }

        writeln!(out, "}}")?;

        stack_fn.extend(func.children);
        fn_index += 1;
    }

    writeln!(out, "}}")?;
    out.flush()?;

    drop(out);

    Ok(std::process::Command::new("dot").arg("output_lifted.dot").arg("-O").arg("-Tsvg").spawn()?)
}

fn output_stage_3_vars(
    top: stage_2::Function,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    let mut lifted = stage_3::Function::lift(top.clone()).with_name("Entrypoint");
    dbg!(lifted.optimize_until_complete());

    let mut out = BufWriter::new(std::fs::File::create("output_lifted_vars.dot")?);

    writeln!(out, "digraph L {{")?;
    writeln!(out, "  graph [splines=ortho, packmode=\"cluster\"]")?;
    writeln!(out, "  node [shape=record fontname=Consolas style=filled color=white margin=0.2];")?;
    let mut fn_index = 0;
    let mut stack_fn = vec![lifted];
    while let Some(func) = stack_fn.pop() {
        writeln!(out, "subgraph cluster_fn_{} {{", fn_index)?;
        writeln!(out, "  fontname = Consolas")?;
        writeln!(out, "  style = filled")?;

        match func.name.as_deref() {
            Some(name) => writeln!(out, "  label = \"{} (Function {})\"", name, fn_index)?,
            None => writeln!(out, "  label = \"Function {}\"", fn_index)?,
        }

        let vars = match func.variables() {
            Some(vars) => vars,
            None => {
                writeln!(out, " fn_{}_warn_lt [label=\"NO_LIFETIMES\"]", fn_index)?;
                vec![]
            },
        };

        // dbg!(&vars);

        let blocks = func.blocks;
        let first_block = blocks.first_key_value().map(|(id, _)| *id);

        for (block_id, block) in blocks {
            let mut content = String::new();
            let mut tooltip = String::new();

            let header = format!("ID {}", *block_id);
            content += &format!("{}\n{}\n", header, "─".repeat(header.chars().count()));

            let pad = (block.instructions.len() as f32).log10() as usize + 1;
            for (idx, instr) in block.instructions.into_iter().enumerate() {
                let mut var_names = HashMap::<Arg, String>::new();

                // TODO: Ridiculously inefficient. Once things work, replace it.
                for var in vars.iter() {
                    let current = (block_id, idx);

                    // Created this block
                    if var.origin == current {
                        var_names.insert(Arg::write(var.arg), format!("var_{}", var.id));
                    }

                    // Used by this block
                    for &location in var.uses.iter() {
                        if location == current {
                            var_names.insert(Arg::read(var.arg), format!("var_{}", var.id));
                        }
                    }

                    // Modified by this block
                    for &location in var.mods.iter() {
                        if location == current {
                            var_names.insert(Arg::modify(var.arg), format!("var_{}", var.id));
                        }
                    }
                }

                let constants = func.constants.as_slice();
                let fallback = |arg: &Arg| arg.text(&constants);
                let text = |arg: &Arg| {
                    var_names
                        .get(arg)
                        .or_else(|| match *arg {
                            // When a variable is merged into another one the origin ArgDir::Write
                            // is stored as a modify event, leading to it later being assumed to be
                            // from an ArgDir::Modify.
                            //
                            // Since an instruction can't both modify and create the same variable,
                            // it is safe to let ArgDir::Modify fallback to ArgDir::Write if it
                            // fails to resolve.
                            Arg(ArgDir::Write, ArgSlot::Register(r)) => {
                                var_names.get(&Arg::modify(ArgSlot::Register(r)))
                            },
                            _ => None,
                        })
                        .cloned()
                        .unwrap_or_else(|| {
                            if arg.is_reg() {
                                log::warn!("fallback: {:?} {} {:?}", block_id, idx, arg);
                            }

                            fallback(arg)
                        })
                };

                let instr_text = instr.text_with(text);
                content += &format!("\u{200B}{: >pad$} | {}\\l", idx, instr_text, pad = pad);
                tooltip += &format!("\u{200B}{: >pad$} | {:?}\\l", idx, instr, pad = pad);
            }

            content = format!(
                "  fn_{}_{} [label=\"{}\", tooltip=\"{}\"]",
                fn_index,
                *block_id,
                escape(content),
                escape(tooltip)
            );

            writeln!(out, "{}", content)?;
            for t in block.forks_to.iter() {
                write!(out, "  fn_{}_{} -> fn_{}_{} ", fn_index, *block_id, fn_index, **t)?;
                match *block_id == **t {
                    true => writeln!(out, "[dir=back]")?,
                    false => writeln!(out)?,
                }
            }
        }

        if !func.children.is_empty() {
            let protos = escape(
                func.children
                    .iter()
                    .enumerate()
                    .map(|(c, v)| format!("{}: {}", c, v.name.as_deref().unwrap_or("✖ unknown ✖")))
                    .join("\\l"),
            );
            writeln!(out, "  protos_{} [label=\"PROTOS\\n──────\\n{}\\l\"]", fn_index, protos)?;

            if let Some(first) = first_block {
                writeln!(out, "protos_{} -> fn_{}_{}", fn_index, fn_index, *first)?;
            }
        }

        writeln!(out, "}}")?;

        stack_fn.extend(func.children);
        fn_index += 1;
    }

    writeln!(out, "}}")?;
    out.flush()?;

    drop(out);

    Ok(std::process::Command::new("dot")
        .arg("output_lifted_vars.dot")
        .arg("-O")
        .arg("-Tsvg")
        .spawn()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let file = std::env::args().nth(1).unwrap_or_else(|| "aram.luabin".to_string());

    log::info!("reading file to bytes");
    let buffer = std::fs::read(file)?;
    let mut cursor = Cursor::new(buffer);

    log::info!("reading LuaFile");
    let chunk = stage_1::LuaFile::read(&mut cursor)?;

    log::info!("parsing top level function");
    let top = stage_2::Function::parse(&chunk.top_level_func);

    log::info!("executing output_stage_1");
    output_stage_1(top.clone());

    log::info!("executing output_stage_2");
    let gv1 = output_stage_2(top.clone())?;

    log::info!("executing output_stage_3");
    let gv2 = output_stage_3(top.clone())?;

    log::info!("executing output_stage_3_vars");
    let gv3 = output_stage_3_vars(top.clone())?;

    log::info!("finished - waiting for graphviz/dot");
    for mut proc in IntoIterator::into_iter([gv1, gv2, gv3]) {
        proc.wait()?;
    }

    Ok(())
}
