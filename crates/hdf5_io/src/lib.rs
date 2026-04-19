use anyhow::Result;
use std::path::Path;

/// Attribute value variants supported for HDF5 group or dataset attributes.
pub enum H5AttrValue {
    String(String),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
}

/// A named attribute to attach to an HDF5 group or dataset.
pub struct H5Attr {
    pub name: String,
    pub value: H5AttrValue,
}

impl H5Attr {
    pub fn string(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self { name: name.into(), value: H5AttrValue::String(value.into()) }
    }

    pub fn u32(name: impl Into<String>, value: u32) -> Self {
        Self { name: name.into(), value: H5AttrValue::U32(value) }
    }

    pub fn i32(name: impl Into<String>, value: i32) -> Self {
        Self { name: name.into(), value: H5AttrValue::I32(value) }
    }

    pub fn f32(name: impl Into<String>, value: f32) -> Self {
        Self { name: name.into(), value: H5AttrValue::F32(value) }
    }

    pub fn f64(name: impl Into<String>, value: f64) -> Self {
        Self { name: name.into(), value: H5AttrValue::F64(value) }
    }
}

/// Writes a slice of attributes to an HDF5 location (group or dataset).
///
/// Both `Group` and `Dataset` deref (possibly through `Container`) to
/// `hdf5::Location`, which owns `new_attr`, so this works for either.
pub fn write_attrs(loc: &hdf5::Location, attrs: &[H5Attr]) -> Result<()> {
    for attr in attrs {
        match &attr.value {
            H5AttrValue::String(s) => {
                let unicode: hdf5::types::VarLenUnicode = s.parse().unwrap();
                loc.new_attr::<hdf5::types::VarLenUnicode>()
                    .shape([1])
                    .create(attr.name.as_str())?
                    .write_raw(&[unicode])?;
            }
            H5AttrValue::U32(v) => {
                loc.new_attr::<u32>().shape([1]).create(attr.name.as_str())?.write_raw(&[*v])?;
            }
            H5AttrValue::I32(v) => {
                loc.new_attr::<i32>().shape([1]).create(attr.name.as_str())?.write_raw(&[*v])?;
            }
            H5AttrValue::F32(v) => {
                loc.new_attr::<f32>().shape([1]).create(attr.name.as_str())?.write_raw(&[*v])?;
            }
            H5AttrValue::F64(v) => {
                loc.new_attr::<f64>().shape([1]).create(attr.name.as_str())?.write_raw(&[*v])?;
            }
        }
    }
    Ok(())
}

/// Opens an existing HDF5 file for read/write, or creates it if absent.
pub fn open_or_create(path: &Path) -> Result<hdf5::File> {
    if path.exists() {
        Ok(hdf5::File::open_rw(path)?)
    } else {
        Ok(hdf5::File::create(path)?)
    }
}

/// Opens or creates a named group at any level of the HDF5 hierarchy.
///
/// `parent` is an `&hdf5::Group`, but since `hdf5::File: Deref<Target=Group>`
/// you can pass a plain `&file` to create a top-level group, or any existing
/// group reference to create a nested subgroup:
///
/// ```ignore
/// let top   = open_or_create_group(&file,  "subject",  force); // top-level
/// let sub   = open_or_create_group(&top?,  "block_0",  false); // nested
/// let deep  = open_or_create_group(&sub?,  "run_1",    false); // deeper
/// ```
///
/// If `force` is `true` and the group already exists it is unlinked first,
/// so the returned group is always empty.
pub fn open_or_create_group(parent: &hdf5::Group, name: &str, force: bool) -> Result<hdf5::Group> {
    match parent.group(name) {
        Ok(existing) => {
            if !force {
                return Ok(existing);
            }
            drop(existing);
            parent.unlink(name)?;
        }
        Err(_) => {}
    }
    // If create_group fails because the name is still present in the symbol table
    // (e.g. a previous run crashed after writing the link but before flushing), fall
    // back to opening the group directly rather than propagating the error.
    match parent.create_group(name) {
        Ok(g) => Ok(g),
        Err(create_err) => parent
            .group(name)
            .map_err(|_| anyhow::anyhow!("cannot create HDF5 group '{}': {}", name, create_err)),
    }
}

/// Writes a typed dataset to an already-open HDF5 group.
///
/// `data` is a flat row-major buffer; `shape` describes the N-dimensional
/// layout (product of dims must equal `data.len()`).  `attrs` are written
/// as dataset-level metadata if provided.
pub fn write_dataset<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    data: &[T],
    shape: &[usize],
    attrs: Option<&[H5Attr]>,
) -> Result<()> {
    let ds = group.new_dataset::<T>().shape(shape).create(name)?;
    ds.write_raw(data)?;
    if let Some(attrs) = attrs {
        write_attrs(&ds, attrs)?;
    }
    Ok(())
}

/// Reads all attributes from an HDF5 location (group or dataset).
///
/// Works on any type that derefs to `hdf5::Location` — pass `&group` or `&dataset`.
/// Attribute types not covered by `H5AttrValue` (compound, fixed strings, etc.)
/// are silently skipped.
pub fn read_attrs(loc: &hdf5::Location) -> Result<Vec<H5Attr>> {
    use hdf5::types::{FloatSize, IntSize, TypeDescriptor};

    let names = loc.attr_names()?;
    let mut attrs = Vec::with_capacity(names.len());

    for name in names {
        let attr = loc.attr(&name)?;
        let desc = attr.dtype()?.to_descriptor()?;

        let value = match desc {
            TypeDescriptor::Unsigned(IntSize::U4) => {
                attr.read_raw::<u32>()?.into_iter().next().map(H5AttrValue::U32)
            }
            TypeDescriptor::Integer(IntSize::U4) => {
                attr.read_raw::<i32>()?.into_iter().next().map(H5AttrValue::I32)
            }
            TypeDescriptor::Float(FloatSize::U4) => {
                attr.read_raw::<f32>()?.into_iter().next().map(H5AttrValue::F32)
            }
            TypeDescriptor::Float(FloatSize::U8) => {
                attr.read_raw::<f64>()?.into_iter().next().map(H5AttrValue::F64)
            }
            TypeDescriptor::VarLenUnicode => attr
                .read_raw::<hdf5::types::VarLenUnicode>()?
                .into_iter()
                .next()
                .map(|s| H5AttrValue::String(s.to_string())),
            _ => None,
        };

        if let Some(value) = value {
            attrs.push(H5Attr { name, value });
        }
    }

    Ok(attrs)
}

/// Reads a typed dataset from an already-open HDF5 group.
///
/// Returns the flat row-major buffer, the dataset shape, and any dataset-level
/// attributes.  Group-level attributes can be read separately with `read_attrs`.
pub fn read_dataset<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
) -> Result<(Vec<T>, Vec<usize>, Vec<H5Attr>)> {
    let ds = group.dataset(name)?;
    let shape = ds.shape();
    let data = ds.read_raw::<T>()?;
    let attrs = read_attrs(&ds)?;
    Ok((data, shape, attrs))
}

/// Opens or creates an HDF5 file and group, then writes a single dataset.
///
/// Useful for simple one-dataset-per-group writes without opening the file
/// manually.  For writing multiple datasets to the same group use
/// `open_or_create` + `open_or_create_group` + `write_dataset` directly.
pub fn append<T: hdf5::H5Type>(
    path: &Path,
    group_name: &str,
    dataset_name: &str,
    data: &[T],
    shape: &[usize],
    group_attrs: Option<&[H5Attr]>,
    dataset_attrs: Option<&[H5Attr]>,
    force: bool,
) -> Result<()> {
    let file = open_or_create(path)?;
    let group = open_or_create_group(&file, group_name, force)?;
    if let Some(attrs) = group_attrs {
        write_attrs(&group, attrs)?;
    }
    write_dataset(&group, dataset_name, data, shape, dataset_attrs)
}

/// Opens an HDF5 file and reads a typed dataset from a named group.
///
/// Returns the flat row-major buffer, the dataset shape, and any dataset-level
/// attributes.  Group-level attributes can be read separately with `read_attrs`
/// after opening the group via `hdf5::File::open` + `file.group(name)`.
pub fn read<T: hdf5::H5Type>(
    path: &Path,
    group_name: &str,
    dataset_name: &str,
) -> Result<(Vec<T>, Vec<usize>, Vec<H5Attr>)> {
    let file = hdf5::File::open(path)?;
    let group = file.group(group_name)?;
    read_dataset(&group, dataset_name)
}
