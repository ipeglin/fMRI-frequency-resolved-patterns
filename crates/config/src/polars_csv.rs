use polars::prelude::*;
use std::{fs, path::Path};

pub fn write_dataframe<P: AsRef<Path>>(file: P, df: &DataFrame) -> PolarsResult<()> {
    let mut file = fs::File::create(&file).expect("could not create file");
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df.to_owned())
}

pub fn write_dataframe_with_file(file: &mut fs::File, df: &DataFrame) -> PolarsResult<()> {
    CsvWriter::new(file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df.to_owned())
}
