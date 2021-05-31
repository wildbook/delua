#![feature(array_windows)]

use std::io::Write;
use std::io::{BufWriter, Cursor};

use itertools::Itertools;

use crate::raw::LuaFile;

mod parsed;
mod raw;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let buffer = std::fs::read("input.luabin")?;
    let mut cursor = Cursor::new(buffer);
    let chunk = LuaFile::read(&mut cursor)?;

    let top = parsed::Function::parse(&chunk.top_level_func);
    let mut stack = vec![top.clone()];
    while let Some(func) = stack.pop() {
        let block = func.blocks();
        println!("|blocks");

        for (_, b) in block.iter().sorted_by_key(|(&a, _)| a) {
            println!("{:#}", b);
        }

        stack.extend(func.prototypes);
    }

    /*
    digraph L {
      node [shape=record fontname=Arial];

      a  [label="one\ltwo three\lfour five six seven\l"]
      b  [label="one\ntwo three\nfour five six seven"]
      c  [label="one\rtwo three\rfour five six seven\r"]

      a -> b -> c
    }
    */

    let mut out = BufWriter::new(std::fs::File::create("output.dot")?);
    writeln!(out, "digraph L {{")?;

    let mut i = 0;
    let mut stack = vec![top.clone()];
    while let Some(func) = stack.pop() {
        writeln!(out, "subgraph fn_{} {{", i)?;
        writeln!(out, "  node [shape=record fontname=Arial];")?;

        let blocks = func.blocks();

        for (a, block) in blocks.iter() {
            let mut code = String::new();
            let mut debug = String::new();
            for instr in block.instructions.iter() {
                code += &format!("{}\\l", instr);
                debug += &format!("{:?}\\l", instr);
            }

            code = format!(
                "  \"fn_{}_{:#x}\" [label={:?}, tooltip={:?}]",
                i, a, code, debug
            );
            code = code.replace('{', r"\{");
            code = code.replace('}', r"\}");
            code = code.replace(r"\\l", r"\l");

            writeln!(out, "{}", code)?;
            for t in block.forks_to.iter() {
                writeln!(
                    out,
                    "\"fn_{}_{:#x}\" -> \"fn_{}_{:#x}\"",
                    i, block.src, i, t.addr
                )?;
            }
        }

        writeln!(out, "}}")?;

        stack.extend(func.prototypes);
        i += 1;
    }

    writeln!(out, "}}")?;
    drop(out);

    std::process::Command::new("dot")
        .arg("output.dot")
        .arg("-O")
        .arg("-Tsvg")
        .spawn()?;

    Ok(())
}
