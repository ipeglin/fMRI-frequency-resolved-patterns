use ndarray::ArrayView2;

/// One connected component in the suprathreshold graph.
pub struct Component {
    /// Upper-triangle edges (i < j) belonging to this component.
    pub edges: Vec<(usize, usize)>,
}

impl Component {
    pub fn size(&self) -> usize {
        self.edges.len()
    }
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        let gp = parent[parent[x]];
        parent[x] = gp;
        x = gp;
    }
    x
}

fn uf_union(parent: &mut [usize], x: usize, y: usize) {
    let rx = uf_find(parent, x);
    let ry = uf_find(parent, y);
    if rx != ry {
        parent[rx] = ry;
    }
}

/// Finds all connected components in the suprathreshold graph defined by `edge_mask`.
/// Only upper-triangle entries (i < j) are considered; the mask is treated as symmetric.
pub fn find_components(edge_mask: ArrayView2<bool>) -> Vec<Component> {
    let c = edge_mask.shape()[0];
    let mut parent: Vec<usize> = (0..c).collect();

    for i in 0..c {
        for j in (i + 1)..c {
            if edge_mask[[i, j]] {
                uf_union(&mut parent, i, j);
            }
        }
    }

    // Collect edges per component root.
    let mut comp_edges: std::collections::HashMap<usize, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    for i in 0..c {
        for j in (i + 1)..c {
            if edge_mask[[i, j]] {
                let root = uf_find(&mut parent, i);
                comp_edges.entry(root).or_default().push((i, j));
            }
        }
    }

    comp_edges
        .into_values()
        .map(|edges| Component { edges })
        .collect()
}

pub fn max_component_size(edge_mask: ArrayView2<bool>) -> usize {
    find_components(edge_mask)
        .iter()
        .map(Component::size)
        .max()
        .unwrap_or(0)
}
