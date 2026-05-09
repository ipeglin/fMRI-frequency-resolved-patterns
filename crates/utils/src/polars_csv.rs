use polars::prelude::*;
use std::{fs, path::Path};

pub fn write_dataframe<P: AsRef<Path>>(file: P, df: &DataFrame) -> PolarsResult<()> {
    write_dataframe_with_separator(file, df, b',')
}

pub fn write_dataframe_with_file(file: &mut fs::File, df: &DataFrame) -> PolarsResult<()> {
    CsvWriter::new(file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df.to_owned())
}

pub fn write_tsv<P: AsRef<Path>>(file: P, df: &DataFrame) -> PolarsResult<()> {
    write_dataframe_with_separator(file, df, b'\t')
}

pub fn write_dataframe_with_separator<P: AsRef<Path>>(
    file: P,
    df: &DataFrame,
    sep: u8,
) -> PolarsResult<()> {
    let mut file = fs::File::create(&file).expect("could not create file");
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(sep)
        .finish(&mut df.to_owned())
}

pub fn read_dataframe<P: AsRef<Path>>(file: P) -> PolarsResult<DataFrame> {
    read_dataframe_with_separator(file, b',')
}

pub fn read_tsv<P: AsRef<Path>>(file: P) -> PolarsResult<DataFrame> {
    read_dataframe_with_separator(file, b'\t')
}

pub fn read_dataframe_with_separator<P: AsRef<Path>>(file: P, sep: u8) -> PolarsResult<DataFrame> {
    let path_str = file
        .as_ref()
        .to_str()
        .ok_or_else(|| polars_err!(ComputeError: "Path is not valid UTF-8"))?;

    let pl_path = PlPath::new(path_str);

    let df = LazyCsvReader::new(pl_path)
        .with_separator(sep)
        .with_has_header(true)
        .with_ignore_errors(true)
        .with_encoding(CsvEncoding::LossyUtf8)
        .finish()?
        .collect()?;

    Ok(df)
}
