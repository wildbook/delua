#![feature(array_windows)]

use std::io::{BufWriter, Cursor, Write};

use itertools::Itertools;

use crate::{
    lifted::{LiftedBlockId, LiftedFunction},
    parsed::OpInfo,
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

fn level_1(top: parsed::Function) {
    let mut stack = vec![top.clone()];
    while let Some(func) = stack.pop() {
        let block = func.blocks();
        println!("|blocks");

        for (_, b) in block.iter().sorted_by_key(|(&a, _)| a) {
            println!("{:#}", b);
        }

        stack.extend(func.prototypes);
    }
}

fn level_2(top: parsed::Function) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = BufWriter::new(std::fs::File::create("output.dot")?);
    writeln!(out, "digraph L {{")?;

    let mut i = 0;
    let mut stack = vec![top.clone()];
    while let Some(func) = stack.pop() {
        writeln!(out, "subgraph fn_{} {{", i)?;
        writeln!(out, "  node [shape=record fontname=Consolas];")?;

        let blocks = func.blocks();

        for (a, block) in blocks.iter() {
            let mut code = String::new();
            let mut debug = String::new();
            for instr in block.instructions.iter() {
                code += &format!("{}\\l", instr);
                debug += &format!("{:?}\\l", instr);
            }

            code = format!(
                "  \"fn_{}_{:#x}\" [label=\"{}\", tooltip=\"{}\"]",
                i,
                a,
                escape(code),
                escape(debug)
            );

            writeln!(out, "{}", code)?;
            for t in block.forks_to.iter() {
                writeln!(
                    out,
                    "\"fn_{}_{:#x}\" -> \"fn_{}_{:#x}\"",
                    i, block.addr_start, i, t.addr
                )?;
            }
        }

        writeln!(out, "}}")?;

        stack.extend(func.prototypes);
        i += 1;
    }

    writeln!(out, "}}")?;
    drop(out);

    std::process::Command::new("dot").arg("output.dot").arg("-O").arg("-Tsvg").spawn()?;

    Ok(())
}

fn level_3(top: parsed::Function) -> Result<(), Box<dyn std::error::Error>> {
    let mut lifted = LiftedFunction::lift(top.clone()).with_name("Entrypoint");

    // Run 10 iterations of optimize and exit early if nothing could be optimized
    for _ in 0..10 {
        if dbg!(lifted.optimize()) == 0 {
            break;
        }
    }

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
        let first_block = Some(LiftedBlockId(0)).filter(|x| blocks.contains_key(x));

        for (block_id, block) in blocks.into_iter() {
            let mut content = String::new();
            let mut tooltip = String::new();

            let header = format!("ID {}", *block_id);
            content += &format!("{}\n{}\n", header, "─".repeat(header.chars().count()));

            for instr in block.instructions.into_iter() {
                content += &format!("{}\\l", instr.text(&func.constants));
                tooltip += &format!("{:?}\\l", instr);
            }

            content = format!(
                "  \"fn_{}_{}\" [label=\"{}\", tooltip=\"{}\"]",
                fn_index,
                *block_id,
                escape(content),
                escape(tooltip)
            );

            writeln!(out, "{}", content)?;
            for t in block.forks_to.iter() {
                writeln!(
                    out,
                    "  \"fn_{}_{}\" -> \"fn_{}_{}\"",
                    fn_index, *block_id, fn_index, **t
                )?;
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
                writeln!(out, "protos_{} -> \"fn_{}_{}\"", fn_index, fn_index, *first)?;
            }
        }

        // if !func.constants.is_empty() {
        //     let consts = escape(
        //         func.constants.iter().enumerate().map(|(c, v)| format!("{}: {}", c, v)).join("\\l"),
        //     );

        //     writeln!(
        //         out,
        //         "  consts_{} [label=\"CONSTANTS\\n─────────\\n{}\\l\"]",
        //         fn_index, consts
        //     )?;

        //     if let Some(first) = first_block {
        //         writeln!(out, "consts_{} -> \"fn_{}_{}\"", fn_index, fn_index, *first)?;
        //     }
        // }

        writeln!(out, "}}")?;

        stack_fn.extend(func.children);
        fn_index += 1;
    }

    writeln!(out, "}}")?;
    drop(out);

    std::process::Command::new("dot")
        .arg("output_lifted.dot")
        .arg("-O")
        .arg("-Tsvg")
        .spawn()?
        .wait()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::args().nth(1).unwrap_or_else(|| "aram.luabin".to_string());
    let buffer = std::fs::read(file)?;
    let mut cursor = Cursor::new(buffer);
    let chunk = LuaFile::read(&mut cursor)?;

    let top = parsed::Function::parse(&chunk.top_level_func);
    level_1(top.clone());
    level_2(top.clone())?;
    level_3(top.clone())?;

    Ok(())
}
