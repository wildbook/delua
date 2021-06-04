#![feature(array_windows)]
#![feature(map_first_last)]

use std::{
    collections::BTreeMap,
    io::{BufWriter, Cursor, Write},
};

use itertools::Itertools;

use crate::{
    lifted::LiftedFunction,
    parsed::{Arg, ExtOpCode, OpInfo},
    raw::LuaFile,
};

mod lifted;
mod parsed;
mod raw;
mod util;

fn escape(s: String) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', r"\n")
        .replace('<', r"\<")
        .replace('>', r"\>")
        .replace('{', r"\{")
        .replace('}', r"\}")
        .replace('"', r#"\""#)
        .replace(r"\\l", r"\l")
}

fn output_parsed_debug(top: parsed::Function) {
    let mut stack = vec![top.clone()];
    while let Some(func) = stack.pop() {
        let block = func.blocks();
        log::debug!("|blocks");

        for (_, b) in block.iter().sorted_by_key(|(&a, _)| a) {
            log::debug!("{:#}", b);
        }

        stack.extend(func.prototypes);
    }
}

fn output_parsed(top: parsed::Function) -> Result<std::process::Child, Box<dyn std::error::Error>> {
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

fn inject_block_lifetime_comments(lifted: &mut LiftedFunction) {
    let mut stack_fn = vec![lifted];
    while let Some(func) = stack_fn.pop() {
        let consts = &func.constants;
        let text = move |arg: Arg| arg.text(consts);

        for (_, block) in func.blocks.iter_mut() {
            let life = match block.lifetime_analysis() {
                Some(life) => life,
                None => {
                    block.push_front([ExtOpCode::comment("LT: ✖ failed ✖")]);
                    continue;
                },
            };

            let mut comments = BTreeMap::<usize, (Vec<String>, Vec<String>)>::new();
            let mut comment = |create, addr, arg| {
                let (c, d) = comments.entry(addr).or_default();
                match create {
                    true => c.push(format!("LT + {}", text(arg))),
                    false => d.push(format!("LT - {}", text(arg))),
                }
            };

            for v in life.vars {
                if !v.arg.is_reg() {
                    continue;
                }

                if let Some(created) = v.created {
                    comment(true, created, v.arg);
                }

                if let Some(deleted) = v.deleted {
                    comment(false, deleted, v.arg);
                }
            }

            let mut offset = 0;
            for (addr, (created, deleted)) in comments {
                for comment in deleted.into_iter().sorted() {
                    let op = ExtOpCode::comment(comment).into();
                    block.instructions.insert(addr + offset, op);
                    offset += 1;
                }

                for comment in created.into_iter().sorted() {
                    let op = ExtOpCode::comment(comment).into();
                    block.instructions.insert(addr + offset, op);
                    offset += 1;
                }
            }

            for import in life.imported.into_iter().filter(Arg::is_reg).sorted() {
                block.push_front([ExtOpCode::comment(format!("LT import {}", text(import)))]);
            }

            for export in life.exported.into_iter().filter(Arg::is_reg).sorted() {
                block.push_back([ExtOpCode::comment(format!("LT export {}", text(export)))]);
            }
        }

        stack_fn.extend(func.children.iter_mut());
    }
}

fn output_lifted(top: parsed::Function) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    let mut lifted = LiftedFunction::lift(top.clone()).with_name("Entrypoint");

    dbg!(lifted.optimize_until_complete());

    inject_block_lifetime_comments(&mut lifted);

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

            for instr in block.instructions.into_iter() {
                content += &format!("{}\\l", instr.text(&func.constants));
                tooltip += &format!("{:?}\\l", instr);
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let file = std::env::args().nth(1).unwrap_or_else(|| "aram.luabin".to_string());

    log::info!("reading file to bytes");
    let buffer = std::fs::read(file)?;
    let mut cursor = Cursor::new(buffer);

    log::info!("reading LuaFile");
    let chunk = LuaFile::read(&mut cursor)?;

    log::info!("parsing top level function");
    let top = parsed::Function::parse(&chunk.top_level_func);

    log::info!("executing output_parsed_debug");
    output_parsed_debug(top.clone());

    log::info!("executing output_parsed");
    let gv1 = output_parsed(top.clone())?;

    log::info!("executing output_lifted");
    let gv2 = output_lifted(top.clone())?;

    log::info!("finished - waiting for graphviz/dot");
    for mut proc in [gv1, gv2] {
        proc.wait()?;
    }

    Ok(())
}
