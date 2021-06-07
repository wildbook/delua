#![feature(array_windows)]
#![feature(map_first_last)]

use std::collections::HashMap;
use std::io::{BufWriter, Cursor, Write};
use std::path::PathBuf;

use itertools::Itertools;
use structopt::StructOpt;

use crate::stage_2::{Arg, ArgDir, ArgSlot, OpInfo};

pub mod stage_1;
pub mod stage_2;
pub mod stage_3;
mod util;

/// This is a terrible `escape` function, but it's good enough for our generated dot output.
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

/// Example function that takes a `stage_2::Function` and outputs some trace info.
fn output_stage_2_trace(func: stage_2::Function) {
    let mut stack = vec![func.clone()];
    while let Some(func) = stack.pop() {
        let block = func.blocks();
        log::trace!("|blocks");

        for (_, b) in block.iter().sorted_by_key(|(&a, _)| a) {
            log::trace!("{:#}", b);
        }

        stack.extend(func.prototypes);
    }
}

/// Example function that takes a `stage_2::Function` and outputs a graph.
fn output_stage_2_graph(
    opt: &Opt,
    func: stage_2::Function,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    let out_path = opt.output_dir.with_file_name("output_s2_graph.dot");
    let mut out = BufWriter::new(std::fs::File::create(&out_path)?);
    writeln!(out, "digraph L {{")?;

    let mut i = 0;
    let mut stack = vec![func.clone()];
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

    Ok(std::process::Command::new("dot").arg(out_path).arg("-O").arg("-Tsvg").spawn()?)
}

/// Example function that takes a `stage_3::Function` and outputs a graph.
fn output_stage_3_graph(
    opt: &Opt,
    func: stage_3::Function,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    let out_path = opt.output_dir.with_file_name("output_s3_graph.dot");
    let mut out = BufWriter::new(std::fs::File::create(&out_path)?);

    writeln!(out, "digraph L {{")?;
    writeln!(out, "  graph [splines=ortho, packmode=\"cluster\"]")?;
    writeln!(out, "  node [shape=record fontname=Consolas style=filled color=white margin=0.2];")?;
    let mut fn_index = 0;
    let mut stack_fn = vec![func];
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

    Ok(std::process::Command::new("dot").arg(out_path).arg("-O").arg("-Tsvg").spawn()?)
}

/// Example function that takes a `stage_3::Function`, tries to recreate variables, then outputs a graph.
fn output_stage_3_vars_graph(
    opt: &Opt,
    func: stage_3::Function,
) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    let out_path = opt.output_dir.with_file_name("output_s3_vars_graph.dot");
    let mut out = BufWriter::new(std::fs::File::create(&out_path)?);

    writeln!(out, "digraph L {{")?;
    writeln!(out, "  graph [splines=ortho, packmode=\"cluster\"]")?;
    writeln!(out, "  node [shape=record fontname=Consolas style=filled color=white margin=0.2];")?;
    let mut fn_index = 0;
    let mut stack_fn = vec![func];
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

                    // Created by this instruction.
                    if var.origin == current {
                        var_names.insert(Arg::write(var.arg), format!("var_{}", var.id));
                    }

                    // Used by this instruction.
                    for &location in var.reads.iter() {
                        if location == current {
                            var_names.insert(Arg::read(var.arg), format!("var_{}", var.id));
                        }
                    }

                    // Modified by this instruction.
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

    Ok(std::process::Command::new("dot").arg(out_path).arg("-O").arg("-Tsvg").spawn()?)
}

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    #[structopt(parse(from_os_str), default_value = "./output/")]
    output_dir: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mut opt = Opt::from_args();

    // Create any missing directories in the output path.
    std::fs::create_dir_all(&opt.output_dir)?;

    // Push a filename, or we'll edit the directory name later.
    opt.output_dir.push("output");

    log::info!("reading file to bytes");
    let buffer = std::fs::read(&opt.input)?;
    let mut cursor = Cursor::new(buffer);

    log::info!("reading LuaFile");
    let s1_file = stage_1::LuaFile::read(&mut cursor)?;
    let s1_func = &s1_file.top_level_func;

    log::info!("parsing top level function");
    let s2_func = stage_2::Function::lift(s1_func);

    log::info!("executing output_stage_1");
    output_stage_2_trace(s2_func.clone());

    log::info!("executing output_stage_2");
    let gv1 = output_stage_2_graph(&opt, s2_func.clone())?;

    let mut s3_func = stage_3::Function::lift(s2_func.clone()).with_name("Entrypoint");
    dbg!(s3_func.optimize_until_complete());

    log::info!("executing output_stage_3");
    let gv2 = output_stage_3_graph(&opt, s3_func.clone())?;

    log::info!("executing output_stage_3_vars");
    let gv3 = output_stage_3_vars_graph(&opt, s3_func)?;

    log::info!("finished - waiting for graphviz/dot");
    for mut proc in IntoIterator::into_iter([gv1, gv2, gv3]) {
        proc.wait()?;
    }

    Ok(())
}
